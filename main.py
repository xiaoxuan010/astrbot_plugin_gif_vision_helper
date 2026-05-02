"""
AstrBot 插件：GIF Vision Helper

功能概述：
- 识别 QQ 传输过来的 GIF 动图（哪怕本地临时文件扩展名是 .jpg/.png，但魔数仍是 GIF）。
- 将 GIF 抽样为多帧 JPEG 静态图：
  - 第一帧覆盖原始 temp 文件路径（兼容 AstrBot 原有调用逻辑）
  - 其余帧追加到 ProviderRequest.image_urls 中，让视觉模型一次看到多帧
- 根据总帧数与文件大小自适应选择抽样帧数，默认最高 6 帧
- 为 LLM 注入一条“GIF 特殊提示”，指明这些图片是从动图抽帧而来
- 维护自己的临时帧文件集合，并在后台异步清理过期文件

部分临时文件管理与提示注入思路参考自：
https://github.com/piexian/astrbot_plugin_gif_to_video
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError  # 新增：单独引入炸弹异常

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register


PLUGIN_ID = "astrbot_plugin_gif_vision_helper"
PLUGIN_VERSION = "0.6.1"
PLUGIN_DESC = "将 QQ 的 GIF 动图转为多帧静态图，让视觉模型真正看懂动图"


@register(PLUGIN_ID, "YanL", PLUGIN_DESC, PLUGIN_VERSION)
class GifVisionHelper(Star):
    """
    GIF → 多帧 JPEG 抽样 + 提示注入 + 临时文件管理
    """

    def __init__(self, context: Context) -> None:
        super().__init__(context)

        # 仅记录本插件创建的临时帧文件，不碰 AstrBot 自己的 temp 管理
        self._temp_files: set[Path] = set()
        self._temp_files_lock = threading.Lock()

        # 轻量缓存：Path -> 创建时间戳，用于过期清理
        self._temp_cache: dict[Path, float] = {}
        # 默认 24 小时 TTL
        self._cache_ttl: float = 24 * 60 * 60

        # Pillow 重采样算法（兼容新旧版本）
        try:
            # Pillow >= 10
            self._resample_lanczos = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except Exception:
            # Pillow < 10 仍然可能存在 Image.LANCZOS；再不行就退到 BICUBIC
            self._resample_lanczos = getattr(Image, "LANCZOS", Image.BICUBIC)

    async def initialize(self) -> None:
        logger.info(
            f"[{PLUGIN_ID}] 插件已初始化"
        )

    async def terminate(self) -> None:
        """
        插件卸载时尽量清理自己注册的临时帧文件。
        """
        logger.info(f"[{PLUGIN_ID}] 插件终止，开始清理临时帧文件 …")
        await asyncio.to_thread(self._cleanup_temp_files)

    # -------------------- 临时文件管理 --------------------

    def _register_temp_file(self, path: Path) -> None:
        """
        仅在本插件自己创建的附加帧文件上调用。
        """
        now = time.time()
        with self._temp_files_lock:
            self._temp_files.add(path)
            self._temp_cache[path] = now

    def _cleanup_temp_files(self) -> None:
        """
        在插件终止时调用：强制清理所有已注册的临时帧文件。
        不操作目录，只删除文件本身，避免误伤共享 temp 目录。
        """
        with self._temp_files_lock:
            paths = list(self._temp_files)

        for p in paths:
            try:
                if p.exists():
                    p.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] 清理临时文件失败 {p}: {e}")
            finally:
                with self._temp_files_lock:
                    self._temp_files.discard(p)
                    self._temp_cache.pop(p, None)

    def _cleanup_expired_cache(self) -> None:
        """
        按 TTL 清理过期的临时帧文件。

        ⚠ 注意：该方法会被放入 asyncio.to_thread 中执行，
        避免在 on_llm_request 里同步阻塞事件循环。
        """
        now = time.time()
        with self._temp_files_lock:
            items = list(self._temp_cache.items())

        for path, ts in items:
            if now - ts <= self._cache_ttl:
                continue

            try:
                if path.exists():
                    path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"[{PLUGIN_ID}] 清理过期临时文件失败 {path}: {e}")
            finally:
                with self._temp_files_lock:
                    self._temp_cache.pop(path, None)
                    self._temp_files.discard(path)

    # -------------------- GIF 检测 + 抽样策略 --------------------

    def _is_gif_file(self, path: Path) -> bool:
        """
        通过魔数判断是否为 GIF。
        即便扩展名是 .jpg/.png，只要 QQ 保留了 GIF 原始二进制，这里仍能识别。
        """
        try:
            with path.open("rb") as f:
                header = f.read(6)
        except FileNotFoundError:
            return False
        except OSError as e:
            logger.warning(f"[{PLUGIN_ID}] 读取文件头失败 {path}: {e}")
            return False

        return header in (b"GIF87a", b"GIF89a")

    def _decide_frame_count(self, total_frames: int, file_size: int) -> int:
        """
        根据总帧数和文件体积，决定抽样帧数。
        """
        if total_frames <= 1:
            return 1

        if total_frames <= 4:
            frames = min(3, total_frames)
        elif total_frames <= 8:
            frames = min(4, total_frames)
        elif total_frames <= 16:
            frames = min(5, total_frames)
        else:
            frames = min(6, total_frames)

        if file_size > 10 * 1024 * 1024:
            frames = max(3, frames - 2)
        elif file_size > 5 * 1024 * 1024:
            frames = max(3, frames - 1)

        return max(1, min(frames, total_frames))

    def _sample_indices(self, n_frames: int, target: int) -> list[int]:
        """
        在 [0, n_frames-1] 区间等间距抽样 target 个帧索引。
        确保包含首尾帧。
        """
        if n_frames <= 0:
            return []
        if target <= 1 or n_frames <= target:
            return list(range(min(n_frames, target)))

        step = (n_frames - 1) / (target - 1)
        indices = sorted({int(round(i * step)) for i in range(target)})
        if not indices:
            return [0]

        indices[0] = 0
        indices[-1] = n_frames - 1
        return indices

    def _resize_frame(self, img: Image.Image, max_side: int) -> Image.Image:
        """
        按最长边等比例缩放到不超过 max_side，使用 LANCZOS（或兼容降级）。
        """
        w, h = img.size
        longest = max(w, h)
        if longest <= max_side:
            return img

        scale = max_side / float(longest)
        new_size = (int(w * scale), int(h * scale))

        try:
            return img.resize(new_size, resample=self._resample_lanczos)
        except Exception as e:
            logger.warning(
                f"[{PLUGIN_ID}] 帧缩放使用 LANCZOS 失败，将使用 Pillow 默认算法: {e}"
            )
            return img.resize(new_size)

    # -------------------- 提示注入 --------------------

    def _inject_preview_hint(self, original_prompt: Optional[str], frame_count: int) -> str:
        """
        为 LLM 注入“这是动图抽帧”的系统提示，尽量不破坏原始 prompt 语义。
        """
        if frame_count > 1:
            hint = (
                f"[系统提示] 本次用户发送的是 GIF 动图，"
                f"插件已将其抽样为 {frame_count} 帧静态图片。"
                f"请综合所有帧理解完整的动作和表情变化，不要只依据第一张图。"
            )
        else:
            hint = (
                "[系统提示] 本次用户发送的是 GIF 动图，"
                "但只成功保留了第一帧静态图片，可能丢失部分动作信息，"
                "请结合上下文和细节进行合理推断。"
            )

        original_prompt = original_prompt or ""
        if hint in original_prompt:
            return original_prompt

        if not original_prompt:
            return hint
        return f"{hint}\n\n{original_prompt}"

    # -------------------- GIF → 多帧 JPEG 核心逻辑 --------------------

    def _convert_gif_to_multi_jpeg(
        self,
        main_path: Path,
        max_side: int,
    ) -> tuple[list[Path], int]:
        """
        核心：把 main_path 指向的 GIF 文件抽样为若干 JPEG 帧。

        ⚠ 重要语义约定：
        - 任何异常（包括 Pillow 的 DecompressionBombError）都会被捕获并记录日志，
          然后返回 ([], 0)，上层将“放行原始请求”，不拦截也不清空图片。
        """
        try:
            file_size = main_path.stat().st_size
        except FileNotFoundError:
            return [], 0

        paths: list[Path] = []

        try:
            with Image.open(main_path) as im:
                total_frames = getattr(im, "n_frames", 1)

                # 非动图：仅保证为 JPEG 并按需缩放
                if total_frames <= 1:
                    rgb = im.convert("RGB")
                    frame0 = self._resize_frame(rgb, max_side)
                    frame0.save(main_path, format="JPEG", quality=90)
                    return [main_path], 1

                target = self._decide_frame_count(total_frames, file_size)
                indices = self._sample_indices(total_frames, target)
                if not indices:
                    return [], 0

                parent = main_path.parent
                stem = main_path.stem
                suffix = ".jpg"

                # 仅缓存第一帧 JPEG 字节，其他帧边读边写，避免按帧累积内存。
                # 第一帧必须最后写回 main_path，防止覆盖原 GIF 后后续 seek 失败。
                first_frame_bytes: Optional[bytes] = None

                for frame_idx in indices:
                    try:
                        im.seek(frame_idx)
                        frame = im.convert("RGB")
                        frame_resized = self._resize_frame(frame, max_side)

                        if first_frame_bytes is None:
                            buffer = io.BytesIO()
                            frame_resized.save(buffer, format="JPEG", quality=90)
                            first_frame_bytes = buffer.getvalue()
                        else:
                            out_path = parent / f"{stem}_f{frame_idx}{suffix}"
                            frame_resized.save(out_path, format="JPEG", quality=90)
                            paths.append(out_path)
                            self._register_temp_file(out_path)
                    except (EOFError, UnidentifiedImageError):
                        # 某些 GIF 标记帧数比实际可读帧数多，跳过异常帧
                        continue

                if first_frame_bytes is None:
                    return [], 0

                # 最后再用第一帧覆盖原始文件，避免覆盖后丢失后续帧数据
                with main_path.open("wb") as f:
                    f.write(first_frame_bytes)
                paths.insert(0, main_path)

                return paths, len(paths)

        except DecompressionBombError as e:
            # 日志里的炸弹异常，现在只打 warning，不再一大坨 error+traceback
            logger.warning(
                f"[{PLUGIN_ID}] GIF 触发 Pillow DecompressionBomb 防御，"
                f"已放弃拆帧并回退到原始请求: {e}"
            )
            return [], 0

        except Exception as e:
            # 其它真正异常仍然视为 error，保留堆栈便于调试
            logger.error(f"[{PLUGIN_ID}] 处理 GIF 时发生异常: {e}", exc_info=True)
            return [], 0

    # -------------------- LLM 请求 Hook --------------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: Any, req: ProviderRequest) -> ProviderRequest:
        """
        在 AstrBot 准备调用视觉 LLM 之前拦截请求。
        """
        _ = event  # 当前版本未使用

        image_urls: Any = getattr(req, "image_urls", None)
        if not image_urls or not isinstance(image_urls, list):
            return req

        first_path = image_urls[0]
        if not isinstance(first_path, str):
            return req

        main_path = Path(first_path)

        # 仅当本地文件头确认为 GIF 时才介入，避免误伤正常 JPEG
        if not self._is_gif_file(main_path):
            return req

        # 异步后台清理过期临时帧文件（不会阻塞当前请求）
        asyncio.create_task(asyncio.to_thread(self._cleanup_expired_cache))

        max_side = 768  # baseline 分辨率上限

        logger.info(
            f"[{PLUGIN_ID}] on_llm_request 捕获到图片请求，尝试处理 GIF: {main_path}"
        )

        try:
            frame_paths, frame_count = await asyncio.to_thread(
                self._convert_gif_to_multi_jpeg,
                main_path,
                max_side,
            )
        except Exception as e:
            # 理论上不会走到这里，因为内部已经吃掉了异常；这里只做兜底日志
            logger.error(
                f"[{PLUGIN_ID}] GIF 转多帧 JPEG 失败（外层兜底），将回退到原始请求: {e}",
                exc_info=True,
            )
            return req

        # 如果没能成功抽出任何帧，就保持行为与之前一致：完全不改 req
        if not frame_paths or frame_count <= 0:
            return req

        # 第一帧覆盖原始路径，其余帧追加到 image_urls
        new_urls: list[str] = [str(frame_paths[0])]
        if len(image_urls) > 1:
            new_urls.extend(str(u) for u in image_urls[1:] if isinstance(u, str))
        for extra in frame_paths[1:]:
            new_urls.append(str(extra))

        try:
            req.image_urls = new_urls  # type: ignore[attr-defined]
        except Exception:
            pass

        current_prompt = getattr(req, "prompt", None)
        try:
            req.prompt = self._inject_preview_hint(current_prompt, frame_count)  # type: ignore[attr-defined]
        except Exception:
            pass

        logger.info(
            f"[{PLUGIN_ID}] GIF 已拆分为 {frame_count} 帧 JPEG，"
            f"image_urls 数量从 {len(image_urls)} → {len(new_urls)}"
        )

        return req
