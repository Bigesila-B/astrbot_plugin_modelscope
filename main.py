import asyncio
import base64
import json
import os
import re
import tempfile
from copy import copy
from io import BytesIO

import aiohttp
from PIL import Image as PILImage

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register

# ── 魔搭社区 API 基础地址 ──
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn"

# ── 支持的比例列表 ──

SUPPORTED_RATIOS = [
    "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3",
    "5:4", "4:5", "21:9", "9:21", "1:2", "2:1",
    "1:3", "3:1", "1:4", "4:1", "1:8", "8:1",
]

RATIO_RE = re.compile(r"^(\d+)[:：](\d+)$")
SIZE_RE = re.compile(r"^\d+x\d+$", re.IGNORECASE)


def _ratio_to_float(ratio_str: str) -> float | None:
    try:
        w_str, h_str = ratio_str.split(":")
        return int(w_str) / int(h_str)
    except (ValueError, ZeroDivisionError):
        return None


def _ratio_to_size(ratio_or_size: str, base: int = 1024) -> str:
    """把比例(如 2:3)转为魔搭 size 像素值(如 768x1152)。

    已是像素值(如 1024x1024)则原样返回；比例按短边 base 推算并对齐到 8 的倍数。
    """
    ratio_or_size = (ratio_or_size or "").strip()
    if not ratio_or_size:
        return f"{base}x{base}"
    # 已是像素值，原样返回
    if SIZE_RE.match(ratio_or_size):
        return ratio_or_size.lower()
    # 中英文冒号统一
    ratio_str = ratio_or_size.replace("：", ":")
    val = _ratio_to_float(ratio_str)
    if val is None:
        return f"{base}x{base}"
    base = max(64, base)
    if val >= 1:
        w, h = base, round(base / val)
    else:
        h, w = base, round(base * val)
    # 对齐到 8 的倍数，且不小于 64
    w = max(64, (w // 8) * 8)
    h = max(64, (h // 8) * 8)
    return f"{w}x{h}"


@register(
    "astrbot_plugin_modelscope",
    "Bigesila-B",
    "魔搭社区(ModelScope)图像生成插件",
    "1.1.2",
    "https://github.com/Bigesila-B/astrbot_plugin_modelscope",
)
class ModelscopePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._models = self._parse_models()
        self._presets = self._parse_presets()
        # 启动日志：便于排查"指令无反应"问题
        logger.info(f"[魔搭插件] 加载完成: 已配置 {len(self._models)} 个生图模型, "
                    f"{len(self._presets)} 个预设")
        for i, m in enumerate(self._models):
            logger.info(f"[魔搭插件] 模型[{i}]: trigger_t2i={m.get('trigger_t2i', '魔搭生图')!r}, "
                        f"trigger_i2i={m.get('trigger_i2i', '魔搭改图')!r}, "
                        f"model_id={m.get('model_id', '?')!r}, "
                        f"apikey={'已填' if m.get('apikey') else '未填'}")
        if not self._models:
            logger.warning("[魔搭插件] ⚠️ 未检测到模型配置！请在 WebUI 插件配置中添加模型并填写 API Key，"
                           "否则发送指令不会有任何反应")
        if self.config.get("enable_natural_language", False):
            for mode in ("t2i", "i2i"):
                _, error = self._natural_language_model(mode)
                if error:
                    logger.warning(f"[魔搭插件] {error}")

    # ── 模型解析 ──

    def _parse_models(self) -> list:
        raw = self.config.get("models", [])
        if isinstance(raw, list):
            return [m for m in raw if isinstance(m, dict) and m.get("model_id")]
        return []

    def _default_model(self) -> dict:
        """返回第一个模型配置作为默认，无配置时返回空 dict"""
        return self._models[0] if self._models else {}

    # ── 自然语言工具调用 ──

    def _natural_language_model(self, mode: str) -> tuple:
        """仅按设置中的模型 ID 匹配已有配置，避免误用其他模型或密钥。"""
        label = "文生图" if mode == "t2i" else "图生图"
        model_id = str(self.config.get(f"natural_language_{mode}_model_id", "") or "").strip()
        if not model_id:
            return None, f"自然语言{label}未配置默认模型 ID，请在插件设置中填写。"
        for model in self._models:
            if str(model.get("model_id", "")).strip() == model_id:
                if not str(model.get("apikey", "") or "").strip():
                    return None, f"自然语言{label}模型「{model_id}」未配置 API Key。"
                return model, ""
        return None, f"自然语言{label}模型「{model_id}」不在生图模型配置中，请先添加对应模型。"

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """按开关和默认模型配置过滤本次请求中的工具，保留其他插件工具。"""
        if req.func_tool is None:
            return
        allowed = set()
        if self.config.get("enable_natural_language", False):
            for mode, name in (("t2i", "modelscope_text_to_image"),
                               ("i2i", "modelscope_image_to_image")):
                model, _ = self._natural_language_model(mode)
                if model:
                    allowed.add(name)
        own_tools = {"modelscope_text_to_image", "modelscope_image_to_image"}
        # ToolSet 可能被其他请求复用；只修改本次请求的副本。
        tool_set = copy(req.func_tool)
        tool_set.tools = [tool for tool in req.func_tool.tools
                          if tool.name not in own_tools or tool.name in allowed]
        req.func_tool = tool_set
        available = {tool.name for tool in tool_set.tools} & allowed
        if available:
            guidance = (
                "\n用户明确要求画图、生成图片或修改图片时，使用可用的 modelscope 生图工具。"
                "从文字生成新图使用 modelscope_text_to_image；编辑用户本条消息或引用消息中的图片"
                "使用 modelscope_image_to_image。仅讨论图片、分析图片或询问绘画方法时不要调用。"
                "图生图工具自行读取消息图片，不要编造图片地址；缺少图片时请用户附图或引用图片。"
                "用户指定的画面要求应完整保留；未指定比例时 aspect_ratio 传空字符串。"
                "工具报告失败时如实说明原因，不要声称已出图，不要自动重复调用。"
            )
            if guidance not in (req.system_prompt or ""):
                req.system_prompt = (req.system_prompt or "") + guidance

    @filter.llm_tool(name="modelscope_text_to_image")
    async def modelscope_text_to_image(self, event: AstrMessageEvent,
                                       prompt: str, aspect_ratio: str = "") -> str:
        """用户明确要求画图或从文字生成新图片时调用，使用设置中的默认文生图模型，并直接发送图片。

        Args:
            prompt(string): 完整的画面描述，保留用户要求，不包含触发词。
            aspect_ratio(string): 用户要求的比例或像素值，如 16:9、1024x1024；未指定时传空字符串。
        """
        return await self._run_natural_language_image(event, prompt, aspect_ratio, "t2i")

    @filter.llm_tool(name="modelscope_image_to_image")
    async def modelscope_image_to_image(self, event: AstrMessageEvent,
                                        prompt: str, aspect_ratio: str = "") -> str:
        """用户明确要求修改、重绘或转换所附图片的风格时调用，使用默认图生图模型，并直接发送图片。

        自动读取本条消息或引用消息中的图片，仅使用第一张；没有图片时不能改图。
        仅识别或分析图片内容时不要调用此工具。

        Args:
            prompt(string): 完整的改图要求，说明需要修改和保留的内容，不包含触发词。
            aspect_ratio(string): 用户要求的比例或像素值，如 16:9、1024x1024；未指定时传空字符串以沿用原图比例。
        """
        return await self._run_natural_language_image(event, prompt, aspect_ratio, "i2i")

    async def _run_natural_language_image(self, event: AstrMessageEvent, prompt: str,
                                          aspect_ratio: str, mode: str) -> str:
        if not self.config.get("enable_natural_language", False):
            return "自然语言生图已关闭，请使用原有触发词指令。"
        model, error = self._natural_language_model(mode)
        if error:
            return error
        if not isinstance(prompt, str) or not prompt.strip():
            return "生图失败：提示词不能为空，请用户补充画面或改图要求。"
        ratio = (aspect_ratio or "").strip() if isinstance(aspect_ratio, str) else ""
        ratio = ratio.replace("：", ":").replace("×", "x").lower()
        if ratio:
            match = re.fullmatch(r"([0-9]+)[:x]([0-9]+)", ratio)
            if not match or any(int(part) <= 0 for part in match.groups()):
                return "生图失败：图片比例或尺寸无效，请使用 16:9 或 1024x1024 等正数格式。"
        try:
            images = []
            if mode == "i2i":
                images = await self._extract_images(event)
                if not images:
                    return "改图失败：本条消息和引用消息中没有可读取的图片，请用户附带或引用一张图片。"
                if not ratio:
                    ratio = await self._get_image_ratio_from_base64(images[0], model)
                images = [self._shrink_for_edit(images[0], max_side=2048)]
            ratio = self._resolve_aspect_ratio(ratio, model)
            action = "处理图片" if mode == "i2i" else "生成图片"
            await event.send(event.plain_result(f"⏳ 正在{action}（{ratio}），请稍候..."))
            result = await self._call_api(prompt.strip(), images=images,
                                          aspect_ratio=ratio, model=model)
            if result.get("status") != "succeeded" or not result.get("results"):
                return f"生图失败：{result.get('error') or '服务未返回图片'}。请如实告知用户，不要自动重试。"
            async for message in self._send_image_result(event, result["results"][0]["url"]):
                await event.send(message)
            return "图片已生成，并已发送给用户。无需重复发图或再次调用生图工具。"
        except Exception:
            logger.exception("[魔搭插件] 自然语言生图处理失败")
            return "生图或图片发送失败，请查看插件日志排查；不要声称图片已成功发送。"

    # ── 预设解析 ──

    def _parse_presets(self) -> list:
        raw = self.config.get("presets", [])
        if isinstance(raw, list):
            return [p for p in raw if isinstance(p, dict) and p.get("trigger")]
        return []

    def _find_preset(self, message_str: str) -> dict | None:
        for preset in self._presets:
            trigger = preset.get("trigger", "")
            if trigger and trigger in message_str:
                return preset
        return None

    def _save_presets(self):
        self.config["presets"] = self._presets
        try:
            self.config.save_config()
        except AttributeError:
            pass

    # ── 消息监听 ──

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，按模型配置的触发词路由"""
        message_str = event.message_str.strip()

        # ── 预设管理命令 ──

        if message_str.startswith("列出预设"):
            async for result in self._handle_list_presets(event):
                yield result
            event.stop_event()
            return

        if message_str.startswith("删除预设"):
            async for result in self._handle_delete_preset(event, message_str):
                yield result
            event.stop_event()
            return

        if message_str.startswith("添加预设"):
            async for result in self._handle_add_preset(event, message_str):
                yield result
            event.stop_event()
            return

        # 优先检查自定义预设触发词（预设触发词更具体，避免被子串误匹配）
        preset = self._find_preset(message_str)
        if preset:
            async for result in self._handle_preset(event, message_str, preset):
                yield result
            event.stop_event()
            return

        # 检查各模型的文生图 / 图生图触发词
        for model in self._models:
            t2i = model.get("trigger_t2i", "魔搭生图").strip()
            i2i = model.get("trigger_i2i", "魔搭改图").strip()

            if t2i and t2i in message_str:
                async for result in self._handle_text_to_image(event, message_str, model, t2i):
                    yield result
                event.stop_event()
                return

            if i2i and i2i in message_str:
                async for result in self._handle_image_edit(event, message_str, model, i2i):
                    yield result
                event.stop_event()
                return

    # ── 预设管理命令 ──

    async def _handle_add_preset(self, event: AstrMessageEvent, message_str: str):
        try:
            payload = message_str.split("添加预设", 1)[1]
        except IndexError:
            yield event.plain_result(
                "格式：添加预设 触发词|LLM地址|APIKey|模型ID|预设提示词\n"
                "示例：添加预设 动漫化|https://api.openai.com/v1|sk-xxx|gpt-4o|分析图片生成动漫风格提示词"
            )
            return
        payload = payload.strip()
        if not payload:
            yield event.plain_result(
                "格式：添加预设 触发词|LLM地址|APIKey|模型ID|预设提示词\n"
                "示例：添加预设 动漫化|https://api.openai.com/v1|sk-xxx|gpt-4o|分析图片生成动漫风格提示词"
            )
            return
        parts = payload.split("|", 4)
        if len(parts) < 5:
            yield event.plain_result("❌ 参数不足，需要 5 个参数（用 | 分隔）")
            return
        trigger, llm_url, llm_apikey, llm_model, preset_prompt = [p.strip() for p in parts]
        if not trigger:
            yield event.plain_result("❌ 触发词不能为空")
            return
        for p in self._presets:
            if p.get("trigger") == trigger:
                yield event.plain_result(f"❌ 触发词「{trigger}」已存在，请先删除再添加")
                return
        self._presets.append({
            "trigger": trigger,
            "llm_url": llm_url,
            "llm_apikey": llm_apikey,
            "llm_model": llm_model,
            "preset_prompt": preset_prompt,
        })
        self._save_presets()
        yield event.plain_result(f"✅ 预设「{trigger}」已添加")

    async def _handle_delete_preset(self, event: AstrMessageEvent, message_str: str):
        try:
            trigger = message_str.split("删除预设", 1)[1].strip()
        except IndexError:
            yield event.plain_result("格式：删除预设 触发词\n示例：删除预设 动漫化")
            return
        if not trigger:
            yield event.plain_result("❌ 请输入要删除的触发词")
            return
        for i, p in enumerate(self._presets):
            if p.get("trigger") == trigger:
                del self._presets[i]
                self._save_presets()
                yield event.plain_result(f"✅ 预设「{trigger}」已删除")
                return
        yield event.plain_result(f"❌ 未找到触发词「{trigger}」的预设")

    async def _handle_list_presets(self, event: AstrMessageEvent):
        if not self._presets:
            yield event.plain_result("📋 当前没有自定义预设")
            return
        lines = ["📋 当前预设："]
        for p in self._presets:
            trigger = p.get("trigger", "?")
            model = p.get("llm_model", "?")
            prompt_preview = p.get("preset_prompt", "")[:40]
            lines.append(f"  • {trigger} → {model} | {prompt_preview}...")
        yield event.plain_result("\n".join(lines))

    # ── 指令处理 ──

    async def _handle_text_to_image(self, event: AstrMessageEvent, message_str: str,
                                    model: dict, trigger: str):
        try:
            text = message_str.split(trigger, 1)[1]
        except IndexError:
            yield event.plain_result(
                f"❌ 请提供生成提示词\n格式：{trigger} [比例] 提示词\n示例：{trigger} 2:3 一只猫")
            return
        prompt, custom_ratio = self._parse_prompt_and_ratio(text)
        if not prompt:
            yield event.plain_result(
                f"❌ 请提供生成提示词\n格式：{trigger} [比例] 提示词\n示例：{trigger} 2:3 一只猫")
            return
        aspect_ratio = self._resolve_aspect_ratio(custom_ratio, model)
        yield event.plain_result(f"⏳ 正在生成图片（{aspect_ratio}），请稍候...")
        result = await self._call_api(prompt, images=[], aspect_ratio=aspect_ratio, model=model)
        if result.get("status") == "succeeded" and result.get("results"):
            async for r in self._send_image_result(event, result["results"][0]["url"]):
                yield r
        else:
            yield event.plain_result(f"❌ 生成失败：{result.get('error') or '未知错误'}")

    async def _handle_image_edit(self, event: AstrMessageEvent, message_str: str,
                                 model: dict, trigger: str):
        try:
            text = message_str.split(trigger, 1)[1]
        except IndexError:
            yield event.plain_result(
                f"❌ 请提供改图提示词和图片\n格式：{trigger} [比例] 提示词 + 图片")
            return
        prompt, custom_ratio = self._parse_prompt_and_ratio(text)
        images = await self._extract_images(event)
        if not prompt:
            yield event.plain_result(
                f"❌ 请提供改图提示词\n格式：{trigger} [比例] 提示词 + 图片")
            return
        if not images:
            yield event.plain_result(
                f"❌ 请附带至少一张图片\n格式：{trigger} [比例] 提示词 + 图片")
            return
        if custom_ratio:
            aspect_ratio = self._resolve_aspect_ratio(custom_ratio, model)
        else:
            img_ratio = await self._get_image_ratio_from_base64(images[0], model)
            if img_ratio:
                aspect_ratio = self._resolve_aspect_ratio(img_ratio, model)
                yield event.plain_result(f"📐 检测到原图比例接近 {img_ratio}，按 {aspect_ratio} 生成")
            else:
                aspect_ratio = self.config.get("aspect_ratio", "1:1")
        yield event.plain_result(
            f"⏳ 正在处理图片（共 {len(images)} 张，{aspect_ratio}），请稍候...")
        # 魔搭 image_url 仅支持单张，取第一张；压缩以符合最长边 ≤ 2048 的限制
        edit_image = self._shrink_for_edit(images[0], max_side=2048)
        result = await self._call_api(
            prompt, images=[edit_image], aspect_ratio=aspect_ratio, model=model)
        if result.get("status") == "succeeded" and result.get("results"):
            async for r in self._send_image_result(event, result["results"][0]["url"]):
                yield r
        else:
            yield event.plain_result(f"❌ 改图失败：{result.get('error') or '未知错误'}")

    # ── 比例解析 ──

    def _parse_prompt_and_ratio(self, text: str) -> tuple:
        text = text.strip()
        if not text:
            return "", None
        parts = text.split(None, 1)
        first_word = parts[0]
        match = RATIO_RE.match(first_word)
        if match:
            ratio = f"{match.group(1)}:{match.group(2)}"
            prompt = parts[1].strip() if len(parts) > 1 else ""
            return prompt, ratio
        return text, None

    def _resolve_aspect_ratio(self, custom_ratio: str | None, model: dict = None) -> str:
        """返回最终使用的比例字符串（如 1:1、16:9）"""
        if not custom_ratio:
            return self.config.get("aspect_ratio", "1:1")
        # 验证比例是否在支持列表中，不在则原样传给 API
        if custom_ratio in SUPPORTED_RATIOS:
            return custom_ratio
        # 可能是像素值如 "1024x1024"，原样传（后续 _ratio_to_size 会识别）
        return custom_ratio

    async def _get_image_ratio_from_base64(self, base64_uri: str, model: dict = None) -> str | None:
        try:
            if "," in base64_uri:
                base64_uri = base64_uri.split(",", 1)[1]
            data = base64.b64decode(base64_uri)
            img = PILImage.open(BytesIO(data))
            w, h = img.size
            if h == 0:
                return None
            return self._find_closest_ratio(w, h, model)
        except Exception as e:
            logger.error(f"解析图片尺寸失败: {e}")
            return None

    def _find_closest_ratio(self, width: int, height: int, model: dict = None) -> str:
        """根据图片宽高，在支持的比例中找到最接近的"""
        target = width / height if height else 1.0
        best_ratio = "1:1"
        best_diff = float("inf")
        for ratio_str in SUPPORTED_RATIOS:
            val = _ratio_to_float(ratio_str)
            if val is None:
                continue
            diff = abs(val - target)
            if diff < best_diff:
                best_diff = diff
                best_ratio = ratio_str
        return best_ratio

    # ── 图片提取 ──

    async def _extract_images(self, event: AstrMessageEvent) -> list:
        return await self._extract_images_from_chain(event.get_messages())

    async def _extract_images_from_chain(self, chain: list) -> list:
        """递归从消息链中提取图片（含引用消息中的图片）"""
        images = []
        for comp in chain:
            if isinstance(comp, Image):
                b64 = await self._image_to_base64_data_uri(comp)
                if b64:
                    images.append(b64)
            elif isinstance(comp, Reply):
                for attr in ("chain", "message", "origin", "content"):
                    nested_chain = getattr(comp, attr, None)
                    if nested_chain and isinstance(nested_chain, list):
                        nested_images = await self._extract_images_from_chain(nested_chain)
                        if nested_images:
                            images.extend(nested_images)
                            break
        logger.info(f"图片提取: 从消息链中提取到 {len(images)} 张图片")
        return images

    async def _image_to_base64_data_uri(self, img: Image) -> str:
        """将 AstrBot Image 组件转为 base64 data URI"""
        # 优先使用官方方法
        try:
            b64 = await img.convert_to_base64()
            if b64:
                return f"data:image/png;base64,{b64}"
        except Exception:
            pass
        # 回退：手动处理 url / file
        if getattr(img, "file", None):
            b64 = await self._file_as_base64(img.file)
            if b64:
                return b64
        if getattr(img, "url", None):
            b64 = await self._download_as_base64(img.url)
            if b64:
                return b64
        logger.warning(f"图片提取: 无法处理 Image 组件, file={getattr(img, 'file', None)}, url={getattr(img, 'url', None)}")
        return ""

    async def _download_as_base64(self, url: str) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.error(f"下载图片失败, status={resp.status}")
                        return ""
                    content_type = resp.headers.get("Content-Type", "image/png")
                    data = await resp.read()
                    b64 = base64.b64encode(data).decode("utf-8")
                    return f"data:{content_type};base64,{b64}"
        except Exception as e:
            logger.error(f"下载图片异常: {e}")
            return ""

    async def _file_as_base64(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            logger.error(f"图片文件不存在: {file_path}")
            return ""
        ext = os.path.splitext(file_path)[1].lower()
        ext_map = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".gif": "gif", ".webp": "webp"}
        mime_type = f"image/{ext_map.get(ext, 'png')}"
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:{mime_type};base64,{b64}"
        except Exception as e:
            logger.error(f"读取图片文件异常: {e}")
            return ""

    # ── 图片压缩（用于 LLM 视觉输入）──

    def _compress_for_llm(self, base64_uri: str) -> str:
        try:
            if "," in base64_uri:
                header, b64 = base64_uri.split(",", 1)
            else:
                header = "data:image/jpeg"
                b64 = base64_uri
            data = base64.b64decode(b64)
            original_size = len(data)
            img = PILImage.open(BytesIO(data))
            w, h = img.size
            needs_resize = w > 1024 or h > 1024
            needs_compress = original_size > 1024 * 1024
            if not needs_resize and not needs_compress:
                return base64_uri
            if needs_resize:
                ratio = 1024.0 / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS)
            if needs_compress or needs_resize:
                quality = 80
                while quality >= 30:
                    buf = BytesIO()
                    img.convert("RGB").save(buf, format="JPEG", quality=quality)
                    if buf.tell() <= 1024 * 1024:
                        break
                    quality -= 10
                buf.seek(0)
                compressed_b64 = base64.b64encode(buf.read()).decode("utf-8")
                logger.info(f"图片压缩: {w}x{h} → {img.size[0]}x{img.size[1]}, "
                            f"{original_size / 1024:.0f}KB → {buf.tell() / 1024:.0f}KB, q={quality}")
                return f"data:image/jpeg;base64,{compressed_b64}"
            return base64_uri
        except Exception as e:
            logger.error(f"图片压缩失败: {e}")
            return base64_uri

    def _shrink_for_edit(self, base64_uri: str, max_side: int = 2048) -> str:
        """压缩图片以符合魔搭图编辑尺寸限制(最长边 ≤ max_side)。

        超限时按比例缩小，统一输出为 JPEG data URI 以避免格式与 header 不匹配。
        """
        try:
            if "," in base64_uri:
                _, b64 = base64_uri.split(",", 1)
            else:
                b64 = base64_uri
            data = base64.b64decode(b64)
            img = PILImage.open(BytesIO(data))
            w, h = img.size
            longest = max(w, h)
            if longest <= max_side:
                return base64_uri  # 尺寸合规，无需处理
            ratio = max_side / longest
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), PILImage.LANCZOS)
            img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=92)
            buf.seek(0)
            new_b64 = base64.b64encode(buf.read()).decode("utf-8")
            logger.info(f"图编辑输入压缩: {w}x{h} → {img.size[0]}x{img.size[1]}, "
                        f"{len(data) / 1024:.0f}KB → {buf.tell() / 1024:.0f}KB")
            return f"data:image/jpeg;base64,{new_b64}"
        except Exception as e:
            logger.error(f"图编辑输入压缩失败: {e}")
            return base64_uri

    # ── 预设工作流 ──

    async def _handle_preset(self, event: AstrMessageEvent, message_str: str, preset: dict):
        trigger = preset["trigger"]
        llm_url = preset.get("llm_url", "").strip()
        llm_apikey = preset.get("llm_apikey", "").strip()
        llm_model = preset.get("llm_model", "gpt-4o").strip()
        preset_prompt = preset.get("preset_prompt", "").strip()
        model_trigger = preset.get("model_trigger", "").strip()

        # LLM 始终需要（无论有无图片，提示词都会经 LLM 优化）
        images = await self._extract_images(event)

        if not llm_url:
            yield event.plain_result(f"❌ 预设「{trigger}」未配置 LLM 地址")
            return
        if not llm_apikey:
            yield event.plain_result(f"❌ 预设「{trigger}」未配置 LLM API Key")
            return
        if not preset_prompt:
            yield event.plain_result(f"❌ 预设「{trigger}」未配置提示词预设")
            return

        # 选择生图模型：按 model_trigger 匹配，否则默认第一个
        gen_model = self._default_model()
        if model_trigger:
            for m in self._models:
                if (m.get("trigger_t2i", "") == model_trigger or
                        m.get("trigger_i2i", "") == model_trigger):
                    gen_model = m
                    break
        if not gen_model:
            yield event.plain_result("❌ 未配置生图模型，请先在插件设置中添加模型")
            return

        try:
            extra_text = message_str.split(trigger, 1)[1].strip()
        except IndexError:
            extra_text = ""

        # 从附加说明开头解析自定义比例（如 "4:3 描述"）
        extra_text, custom_ratio = self._parse_prompt_and_ratio(extra_text)

        # 无图片：仍经 LLM 优化提示词后文生图（不传图片给 LLM）
        if not images:
            if not extra_text:
                yield event.plain_result(
                    f"❌ 请提供提示词或附带图片\n格式：{trigger} [比例] 提示词\n或：{trigger} [比例] [附加说明] + 图片")
                return
            llm_prompt = f"{preset_prompt}\n\n用户附加要求：{extra_text}" if extra_text else preset_prompt
            yield event.plain_result(f"⏳ 正在生成「{trigger}」，请稍候...")

            llm_result = await self._call_llm_vision(llm_url, llm_apikey, llm_model, llm_prompt, [])
            if llm_result.startswith("LLM 错误") or llm_result.startswith("LLM 调用失败"):
                yield event.plain_result(f"❌ {llm_result}")
                return
            if not llm_result or not llm_result.strip():
                yield event.plain_result("❌ 语言模型未返回有效提示词，请检查 LLM 配置")
                return

            aspect_ratio = self._resolve_aspect_ratio(custom_ratio, gen_model)
            logger.info(f"[魔搭插件] 预设文生图(无图): custom_ratio={custom_ratio}, "
                        f"aspect_ratio={aspect_ratio}, model={gen_model.get('model_id')}")
            result = await self._call_api(
                llm_result, images=[], aspect_ratio=aspect_ratio, model=gen_model)
            if result.get("status") == "succeeded" and result.get("results"):
                async for r in self._send_image_result(event, result["results"][0]["url"]):
                    yield r
            else:
                yield event.plain_result(f"❌ 生成失败：{result.get('error') or '未知错误'}")
            return

        # 有图片：走 LLM 视觉分析 → 生图
        llm_prompt = f"{preset_prompt}\n\n用户附加要求：{extra_text}" if extra_text else preset_prompt
        llm_images = [self._compress_for_llm(img) for img in images]

        yield event.plain_result(f"⏳ 正在生成「{trigger}」，请稍候...")

        llm_result = await self._call_llm_vision(llm_url, llm_apikey, llm_model, llm_prompt, llm_images)

        if llm_result.startswith("LLM 错误") or llm_result.startswith("LLM 调用失败"):
            yield event.plain_result(f"❌ {llm_result}")
            return
        if not llm_result or not llm_result.strip():
            yield event.plain_result("❌ 语言模型未返回有效提示词，请检查 LLM 配置或图片是否可识别")
            return

        # 比例优先级：用户指定 > 原图比例 > 默认
        if custom_ratio:
            aspect_ratio = self._resolve_aspect_ratio(custom_ratio, gen_model)
        else:
            img_ratio = await self._get_image_ratio_from_base64(images[0])
            aspect_ratio = self._resolve_aspect_ratio(img_ratio) if img_ratio else self.config.get("aspect_ratio", "1:1")
        logger.info(f"[魔搭插件] 预设比例: custom_ratio={custom_ratio}, "
                    f"aspect_ratio={aspect_ratio}, model={gen_model.get('model_id')}")

        # 调用魔搭生图模型出图（预设工作流通常为文生图，不传输入图片）
        result = await self._call_api(
            llm_result, images=[], aspect_ratio=aspect_ratio, model=gen_model)

        if result.get("status") == "succeeded" and result.get("results"):
            async for r in self._send_image_result(event, result["results"][0]["url"]):
                yield r
        else:
            yield event.plain_result(f"❌ 生成失败：{result.get('error') or '未知错误'}")

    def _parse_llm_raw_response(self, raw: str) -> dict | None:
        """解析 LLM 返回的异常格式响应，尽量提取出有效内容。

        支持：多个 JSON 对象拼接（}{相连）、SSE 流式（data: 前缀）。
        返回标准化的 {"choices": [{"message": {"content": "..."}}]}，
        或 {"_empty": True, "_count": N, "_null_choices": M} 表示解析到 N 个对象
        但无内容（其中 M 个 choices 为 null），或 None 表示完全无法解析。
        """
        import json as _json
        combined_text = ""
        parsed_count = 0
        null_choices_count = 0

        # 1) 尝试 SSE 流式格式（data: 前缀）
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                chunk_str = line[5:].strip()
                if not chunk_str or chunk_str == "[DONE]":
                    continue
                try:
                    chunk = _json.loads(chunk_str)
                    parsed_count += 1
                    choices = chunk.get("choices")
                    if choices is None:
                        null_choices_count += 1
                    elif choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content") or ""
                        if text:
                            combined_text += text
                except _json.JSONDecodeError:
                    continue

        # 2) 尝试多个 JSON 对象拼接（用 JSONDecoder 逐个解析）
        if not combined_text:
            decoder = _json.JSONDecoder()
            idx = 0
            raw_stripped = raw.strip()
            while idx < len(raw_stripped):
                # 跳过空白和分隔符
                while idx < len(raw_stripped) and raw_stripped[idx] in " \t\n\r":
                    idx += 1
                if idx >= len(raw_stripped):
                    break
                try:
                    obj, end_idx = decoder.raw_decode(raw_stripped, idx)
                    idx = end_idx
                    if isinstance(obj, dict):
                        parsed_count += 1
                        choices = obj.get("choices")
                        if choices is None:
                            null_choices_count += 1
                        elif choices:
                            msg = choices[0].get("message", {}) or choices[0].get("delta", {})
                            text = msg.get("content") or ""
                            if text:
                                combined_text += text
                except _json.JSONDecodeError:
                    break

        if combined_text:
            return {"choices": [{"message": {"content": combined_text}}]}
        if parsed_count > 0:
            return {"_empty": True, "_count": parsed_count, "_null_choices": null_choices_count}
        return None

    async def _call_llm_vision(self, llm_url: str, apikey: str, model: str,
                               prompt: str, images: list) -> str:
        url = llm_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {apikey}", "Content-Type": "application/json"}
        content = [{"type": "text", "text": prompt}]
        for img_b64 in images:
            content.append({"type": "image_url", "image_url": {"url": img_b64}})
        payload = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": 2048}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    raw = await resp.text()
                    # 尝试解析 JSON
                    try:
                        result = json.loads(raw)
                    except json.JSONDecodeError:
                        # 可能是多个 JSON 对象拼接、SSE 流式响应等，逐段解析
                        result = self._parse_llm_raw_response(raw)
                        if result is None:
                            logger.error(f"LLM 响应非JSON，前300字符: {raw[:300]}")
                            return f"LLM 错误：响应格式异常(非JSON)，HTTP {resp.status}"
                        # 解析到 JSON 但内容为空（choices=null）
                        if result.get("_empty"):
                            count = result.get("_count", 0)
                            null_count = result.get("_null_choices", 0)
                            logger.error(
                                f"LLM 返回 {count} 个响应但内容均为空"
                                f"（其中 {null_count} 个 choices=null），"
                                f"model={model}, 前300字符: {raw[:300]}")
                            return (f"LLM 错误：服务端返回 {count} 个空响应"
                                    f"（{null_count} 个 choices=null），"
                                    f"请检查 API Key、模型 ID（{model}）是否正确或配额是否充足")
                    logger.info(f"LLM API 响应: has_choices={'choices' in result}, "
                                f"model={model}, images={len(images)}")
                    if "choices" in result and len(result["choices"]) > 0:
                        msg = result["choices"][0].get("message", {})
                        content = (msg.get("content") or
                                   msg.get("reasoning_content") or "")
                        content = content.strip()
                        if content:
                            return content
                        logger.warning(f"LLM 返回空内容: choices={result['choices'][0]}")
                        return "LLM 错误：模型未返回有效文本（可能拒绝处理或内容被过滤）"
                    err = result.get("error", {})
                    err_msg = err.get("message", "") if isinstance(err, dict) else str(err)
                    return f"LLM 错误：{err_msg or '未知错误'}"
        except aiohttp.ClientError as e:
            logger.error(f"LLM 请求异常: {e}")
            return f"LLM 调用失败：{e}"
        except Exception as e:
            logger.error(f"LLM 调用异常: {e}")
            return f"LLM 调用失败：{e}"

    # ── 图片发送 ──

    async def _send_image_result(self, event: AstrMessageEvent, image_url: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.error(f"下载生成图片失败, status={resp.status}")
                        yield event.image_result(image_url)
                        return
                    data = await resp.read()
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(data)
                        temp_path = f.name
                    yield event.chain_result([Image.fromFileSystem(temp_path)])
        except Exception as e:
            logger.error(f"发送图片异常: {e}")
            yield event.image_result(image_url)

    # ── 魔搭社区 API 调用 ──

    async def _call_api(self, prompt: str, images: list, aspect_ratio: str = None,
                        model: dict = None) -> dict:
        """调用魔搭社区 API-Inference 文生图/图编辑接口。

        返回统一格式：
          成功 {"status": "succeeded", "results": [{"url": ...}]}
          失败 {"status": "failed", "error": "..."}
        """
        model = model or self._default_model()
        apikey = model.get("apikey", "").strip()
        model_id = model.get("model_id", "krea/Krea-2-Turbo").strip()
        negative_prompt = model.get("negative_prompt", "").strip()
        base_url = model.get("url", "").strip() or MODELSCOPE_BASE_URL
        poll_timeout = model.get("poll_timeout", 180)
        if not isinstance(poll_timeout, (int, float)) or poll_timeout <= 0:
            poll_timeout = 180
        size_base = model.get("size_base", 1024)
        try:
            size_base = int(size_base)
        except (ValueError, TypeError):
            size_base = 1024
        if size_base < 64:
            size_base = 64

        if not apikey:
            return {"status": "failed", "error": "未配置 API Key"}
        if not prompt or not prompt.strip():
            return {"status": "failed", "error": "提示词为空，无法生成图片"}

        # 比例/像素 → 魔搭 size（按模型配置的短边基础像素推算）
        if aspect_ratio is None:
            aspect_ratio = self.config.get("aspect_ratio", "1:1")
        size = _ratio_to_size(aspect_ratio, size_base)
        logger.info(f"[魔搭插件] 调用API: model={model_id}, size={size}, "
                    f"aspect_ratio={aspect_ratio}, size_base={size_base}, has_image={bool(images)}")

        headers = {
            "Authorization": f"Bearer {apikey}",
            "Content-Type": "application/json",
            "X-ModelScope-Async-Mode": "true",
        }
        payload = {"model": model_id, "prompt": prompt, "size": size}
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        # 魔搭图编辑：image_url 单张（base64 data URI 或公网 URL），仅编辑模型支持
        if images:
            payload["image_url"] = images[0]

        try:
            async with aiohttp.ClientSession() as session:
                # 1. 提交异步生图任务
                async with session.post(
                    f"{base_url.rstrip('/')}/v1/images/generations",
                    json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        return {"status": "failed",
                                "error": f"提交失败({resp.status}): {err_text}"}
                    data = await resp.json()
                    task_id = data.get("task_id")
                    if not task_id:
                        return {"status": "failed",
                                "error": f"未返回 task_id: {data}"}
                    logger.info(f"魔搭 API: 已提交任务 task_id={task_id}, "
                                f"model={model_id}, size={size}, imgs={len(images)}, "
                                f"prompt_len={len(prompt)}")
                # 2. 轮询任务结果
                return await self._poll_task(apikey, task_id, base_url, poll_timeout)
        except aiohttp.ClientError as e:
            logger.error(f"魔搭 API 请求异常: {e}")
            return {"status": "failed", "error": f"网络请求失败：{e}"}
        except Exception as e:
            logger.error(f"魔搭 API 调用异常: {e}")
            return {"status": "failed", "error": f"调用失败：{e}"}

    async def _poll_task(self, apikey: str, task_id: str, base_url: str,
                         timeout_sec: float = 180) -> dict:
        """轮询魔搭任务状态，返回统一格式的结果字典"""
        headers = {
            "Authorization": f"Bearer {apikey}",
            "X-ModelScope-Task-Type": "image_generation",
        }
        interval = 3.0
        max_retries = max(1, int(timeout_sec / interval))
        poll_url = f"{base_url.rstrip('/')}/v1/tasks/{task_id}"
        try:
            async with aiohttp.ClientSession() as session:
                for attempt in range(max_retries):
                    await asyncio.sleep(interval)
                    try:
                        async with session.get(
                            poll_url, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=15),
                        ) as resp:
                            data = await resp.json()
                            status = data.get("task_status")
                            if status == "SUCCEED":
                                imgs = data.get("output_images") or []
                                if imgs:
                                    return {"status": "succeeded",
                                            "results": [{"url": imgs[0]}]}
                                return {"status": "failed",
                                        "error": "任务成功但未返回图片"}
                            if status == "FAILED":
                                return {"status": "failed",
                                        "error": "生成失败(任务返回 FAILED)"}
                            # RUNNING / PENDING 等状态继续轮询
                    except Exception as e:
                        logger.error(f"轮询异常 (尝试 {attempt + 1}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"轮询会话异常: {e}")
        return {"status": "failed", "error": f"任务超时（{timeout_sec:.0f} 秒）"}

    async def terminate(self):
        pass
