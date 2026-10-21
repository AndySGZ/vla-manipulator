"""Report cards: one PNG per level, ready to drop into a slide deck.

Each card carries, in a fixed layout:

    1. the policy's view, clean            (the control side)
    2. the policy's view, underwater       (same seed, same layout)
    3. a third-person view                 (so a human can see the scene)

    plus the difficulty axis, the object, the goal, the criterion, the measured
    obstacle numbers, the instruction pools, and a footer that names the spec
    fingerprint and whether the scene self-test passed.

Only panels 1 and 2 are what the policy sees; panel 3 exists because "the
policy's view" is a 224x224 top-down frame in which a guide wall and a cube look
alike, and a reader needs to be shown what the level actually is.

The footer is the part that makes the card trustworthy rather than decorative:
it prints the ``spec_hash`` and the self-test result, so a number on the card can
be traced back to the code that produced it.

    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.shots
    PYTHONPATH=<WYSD> uv run python -m mj_env.scripts.shots --task o4_narrow_slot
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mj_env.scene import SCENE_CAMERA
from mj_env.specs import LEVEL_ORDER, get
from mj_env.task_env import TaskEnv
from mj_env.vfx import CLEAN_RUNG, LADDER, Imaging, load_rungs

DEFAULT_OUT = Path("shots")
SELFTEST_REPORT = Path(".tmp/selftest/selftest_report.json")
WATER_RUNG = "v2_medium"

VERSION = "v0.1"

# --- look ------------------------------------------------------------------
BG = (18, 20, 24)
PANEL_BG = (28, 31, 36)
HEADER_BG = (30, 34, 42)
RULE = (60, 66, 76)
TITLE = (238, 240, 244)
LABEL = (138, 148, 164)
VALUE = (216, 222, 232)
ACCENT = (86, 156, 214)
WARN = (214, 152, 86)
OK = (126, 190, 122)

#: CJK-capable fonts, first hit wins.  A missing font would silently render every
#: Chinese label as boxes, so the search is explicit and the fallback is loud.
FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\Deng.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

CARD_WIDTH = 1136
MARGIN = 24
GUTTER = 12
PANEL_SCALE = 1.5
ROW_GAP = 10


@dataclass
class Fonts:
    title: ImageFont.FreeTypeFont
    label: ImageFont.FreeTypeFont
    value: ImageFont.FreeTypeFont
    small: ImageFont.FreeTypeFont
    mono: ImageFont.FreeTypeFont

    @classmethod
    def load(cls) -> "Fonts":
        path = next((candidate for candidate in FONT_CANDIDATES if Path(candidate).is_file()), None)
        if path is None:
            raise SystemExit(
                "no CJK font found; cards would render Chinese as boxes. "
                f"looked for: {FONT_CANDIDATES}"
            )
        mono_path = next(
            (
                candidate
                for candidate in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\lucon.ttf")
                if Path(candidate).is_file()
            ),
            path,
        )
        return cls(
            title=ImageFont.truetype(path, 30),
            label=ImageFont.truetype(path, 17),
            value=ImageFont.truetype(path, 19),
            small=ImageFont.truetype(path, 15),
            mono=ImageFont.truetype(mono_path, 15),
        )


def wrap(text: str, font, width: int) -> list[str]:
    """Character-wise wrap.  CJK has no spaces, so word wrapping is not an option."""
    lines: list[str] = []
    current = ""
    for character in text:
        if character == "\n":
            lines.append(current)
            current = ""
            continue
        if font.getlength(current + character) > width and current:
            lines.append(current)
            current = character
        else:
            current += character
    if current:
        lines.append(current)
    return lines or [""]


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_panel(
    env: TaskEnv,
    camera: str | int,
    *,
    size: int,
    rung: str = CLEAN_RUNG,
    free_view: tuple[float, float, float] | None = None,
    lookat: np.ndarray | None = None,
) -> np.ndarray:
    """One square frame: a named camera through a rung, or a free-camera view."""
    if free_view is not None:
        renderer = mujoco.Renderer(env.model, size, size)
        try:
            # `scene.camera` is a read-only tuple in this MuJoCo version, so the
            # free camera is built separately and handed to `update_scene`, which
            # accepts an `MjvCamera` directly.
            camera_object = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(env.model, camera_object)
            camera_object.lookat[:] = lookat if lookat is not None else (0.2, 0.0, 0.04)
            camera_object.distance, camera_object.azimuth, camera_object.elevation = free_view
            renderer.update_scene(env.data, camera=camera_object)
            return np.asarray(renderer.render(), dtype=np.uint8).copy()
        finally:
            renderer.close()
    imaging = Imaging(env.model, size, size, rung)
    try:
        return imaging.render(env.data, camera)
    finally:
        imaging.close()


def level_panels(env: TaskEnv, *, size: int, water_rung: str) -> list[np.ndarray]:
    """The three panels, from the reset pose -- the state an episode starts in."""
    lookat = np.asarray(env.prop_pos(env.target_object), dtype=float)
    lookat[2] = 0.04
    return [
        render_panel(env, SCENE_CAMERA, size=size),
        render_panel(env, SCENE_CAMERA, size=size, rung=water_rung),
        render_panel(env, None, size=size, free_view=(0.62, 165.0, -22.0), lookat=lookat),
    ]


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------
def obstacle_rows(env: TaskEnv, spec) -> list[tuple[str, str]]:
    """Measured geometry of everything static except the goal carrier.

    The offset is reported **in the target's own frame**, because that is the
    quantity the level is defined by: it is what stays constant across seeds
    (the scene self-test asserts exactly this), whereas the world position
    changes every reset.
    """
    rows = []
    target_body = env._objects[spec.target].body_id
    rotation = env.data.xmat[target_body].reshape(3, 3)
    for name in spec.statics:
        if name == spec.goal_body:
            continue
        body = env._static_ids[name]
        geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        world = np.asarray(env.model.body_pos[body], dtype=float)
        local = rotation.T @ (world - env.prop_pos(spec.target))
        if geom < 0:
            # A marker body with no geometry (O5's parking spot).  Reporting
            # ``model.geom_size[-1]`` here prints a size that belongs to some
            # other geom -- and a wrong number on a card is indistinguishable
            # from a measured one, which is exactly what a card must not do.
            rows.append(
                (
                    f"标记 · {name}",
                    "无几何的标记体（不参与碰撞、不进相机）；位置随 reset 变化，"
                    f"本次位于目标自身坐标系 "
                    f"({local[0] * 1000:+.0f}, {local[1] * 1000:+.0f}) mm",
                )
            )
            continue
        size = np.asarray(env.model.geom_size[geom], dtype=float) * 1000.0
        rows.append(
            (
                f"障碍物 · {name}",
                f"盒子 {size[0] * 2:.0f}×{size[1] * 2:.0f}×{size[2] * 2:.0f} mm；"
                f"在目标自身坐标系里偏 ({local[0] * 1000:+.0f}, {local[1] * 1000:+.0f},"
                f" {local[2] * 1000:+.0f}) mm —— 每次 reset 随目标一起移动并旋转",
            )
        )
    return rows


def selftest_rows() -> dict[str, str]:
    """Per-level self-test status, straight from the last run's report."""
    if not SELFTEST_REPORT.is_file():
        return {}
    payload = json.loads(SELFTEST_REPORT.read_text(encoding="utf-8"))
    status = {}
    for task_id, results in payload.get("levels", {}).items():
        failed = [row for row in results if not row["ok"]]
        status[task_id] = (
            f"{len(results)} 项全绿" if not failed else f"{len(failed)}/{len(results)} 项失败"
        )
    return status


def build_card(
    task_id: str,
    panels: list[np.ndarray],
    rows: list[tuple[str, str]],
    fonts: Fonts,
    *,
    footer: str,
) -> Image.Image:
    spec = get(task_id)
    family_levels = [key for key in LEVEL_ORDER if get(key).family == spec.family]
    header_text = f"{spec.title}"
    header_right = f"族 {spec.family} · 等级 {family_levels.index(task_id) + 1}/{len(family_levels)}"

    inner = CARD_WIDTH - 2 * MARGIN
    panel = int(panels[0].shape[0] * PANEL_SCALE)
    panel_row = panel + 22

    # Measure the text block before allocating the canvas: a card whose height is
    # guessed either crops the last row or leaves a hole.
    wrapped: list[tuple[str, list[str]]] = []
    for label, value in rows:
        wrapped.append((label, wrap(value, fonts.value, inner - 190)))
    text_height = sum(len(lines) * 26 + ROW_GAP for _, lines in wrapped)

    header_height = 64
    footer_height = 46
    height = header_height + panel_row + text_height + footer_height + 2 * MARGIN

    image = Image.new("RGB", (CARD_WIDTH, height), BG)
    draw = ImageDraw.Draw(image)

    # header
    draw.rectangle([0, 0, CARD_WIDTH, header_height], fill=HEADER_BG)
    badge = f"{spec.family}{spec.level}"
    draw.rounded_rectangle([MARGIN, 16, MARGIN + 52, 48], radius=6, fill=ACCENT)
    draw.text((MARGIN + 12, 22), badge, font=fonts.label, fill=(12, 16, 22))
    draw.text((MARGIN + 68, 20), header_text, font=fonts.title, fill=TITLE)
    draw.text(
        (CARD_WIDTH - MARGIN - fonts.label.getlength(header_right), 26),
        header_right,
        font=fonts.label,
        fill=LABEL,
    )

    # panels
    y = header_height + 12
    x = MARGIN
    captions = ("① 策略视角 · 无水下", f"② 策略视角 · {WATER_RUNG}", "③ 第三人称")
    for panel_image, caption in zip(panels, captions):
        scaled = Image.fromarray(panel_image).resize((panel, panel), Image.BILINEAR)
        image.paste(scaled, (x, y + 20))
        draw.text((x, y), caption, font=fonts.small, fill=LABEL)
        x += panel + GUTTER

    # text rows
    y += panel_row
    for label, lines in wrapped:
        draw.text((MARGIN, y + 2), label, font=fonts.label, fill=LABEL)
        for index, line in enumerate(lines):
            draw.text((MARGIN + 190, y + index * 26), line, font=fonts.value, fill=VALUE)
        y += len(lines) * 26 + ROW_GAP

    # footer
    draw.line([MARGIN, height - footer_height, CARD_WIDTH - MARGIN, height - footer_height], fill=RULE)
    # A CJK font, not the mono one: Consolas has no Chinese glyphs and the labels
    # silently render as boxes, which is exactly the kind of thing that gets
    # noticed on a projector rather than here.
    draw.text((MARGIN, height - 32), footer, font=fonts.small, fill=LABEL)
    return image


def footer_for(task_id: str, env: TaskEnv, water_rung: str, status: str) -> str:
    spec = get(task_id)
    rungs = load_rungs()
    visibility = (
        f"{rungs[water_rung].visibility_m:.2f}m"
        if water_rung in rungs and rungs[water_rung].visibility_m
        else "n/a"
    )
    return (
        f"{VERSION} · spec_hash={spec.hash()} · 水下档位 {water_rung} ({visibility})"
        f" · 场景自检：{status} · 判据与专家由 mj_env.test_criteria / selftest_scenes 保证"
    )


def plain(text: str) -> str:
    """Strip markdown emphasis: the specs are written for the docs, not for PIL."""
    return text.replace("**", "").replace("`", "")


def characteristics(env: TaskEnv, spec, status: str) -> list[tuple[str, str]]:
    target = env.target_object
    width = env._objects[target].grip_width * 1000.0
    goal = env._goal_body_id
    goal_desc = "无"
    if goal is not None:
        geom = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, f"{spec.goal_body}_geom"
        )
        radius = np.asarray(env.model.geom_size[geom], dtype=float)[0] * 1000.0
        goal_desc = f"{spec.goal_body}（φ{radius * 2:.0f} mm 圆盘，无碰撞；方块落在台面上）"

    notes = spec.notes.strip().replace("\n", " ")
    if len(notes) > 96:
        notes = notes[:96] + "…"
    rows = [
        ("难度轴", plain(spec.difficulty_axis)),
        ("被操作物", f"{', '.join(spec.prop_bodies)}；本段目标 {target}，握宽 {width:.1f} mm"),
        ("目标区", goal_desc),
        ("成功判据", f"{plain(spec.criterion_text)}（hold {spec.hold_s}s）"),
        ("观测 / 动作", "两路 224² RGB + 6 维状态；动作 6 维（5 关节 + 夹爪），20 Hz —— schema 冻结"),
        ("水下对比", f"{CLEAN_RUNG} 与 {WATER_RUNG}（v1~v3 见 configs/vfx/so101_tabletop.json）"),
        ("场景文件", f"{spec.scene}（由 assets/base_table.xml 合成）"),
        ("指令池", f"{len(spec.instructions)} 训练 + {len(spec.holdout)} hold-out（只评测、不进训练）"),
        ("实测", f"场景自检 {status}；{notes}"),
    ]
    rows.extend(obstacle_rows(env, spec))
    return rows


def build_overview(
    entries: list[tuple[str, list[np.ndarray], str]], *, size: int, status: dict[str, str]
) -> Image.Image:
    """One row per level: the clean/water pair, the difficulty axis, the criterion.

    This is the single figure that explains the ladder, so every row carries the
    three things a viewer asks in order: what changes, what has to be true, and
    is it verified.
    """
    thumb = int(size * 1.15)
    text_x = MARGIN + 2 * thumb + 24
    # The criterion is drawn indented under its own label, so the wrap width has
    # to leave room for that indent -- otherwise the last characters run off the
    # right edge of the card.
    text_width = 1136 - MARGIN - text_x - 46
    row_height = thumb + 30
    fonts = Fonts.load()

    parts = {
        task_id: {
            "criterion": wrap(plain(get(task_id).criterion_text), fonts.small, text_width),
            "status": status.get(task_id, "未运行"),
        }
        for task_id, _, _ in entries
    }
    image = Image.new("RGB", (1136, 62 + len(entries) * row_height), BG)
    draw = ImageDraw.Draw(image)
    total = sum(1 for task_id, _, _ in entries if status.get(task_id, "").endswith("全绿"))
    draw.text(
        (MARGIN, 16),
        f"SO101 场景阶梯 · {len(entries)} 级 · clean / underwater 成对（同 seed、同布局）"
        f" · 自检全绿的级数 {total}/{len(entries)}",
        font=fonts.value,
        fill=TITLE,
    )
    for index, (task_id, panels, axis) in enumerate(entries):
        spec = get(task_id)
        y = 62 + index * row_height
        draw.text((MARGIN, y + 2), f"{spec.family}{spec.level}", font=fonts.label, fill=ACCENT)
        draw.text((MARGIN + 56, y + 2), spec.title, font=fonts.value, fill=TITLE)
        draw.text((MARGIN + 240, y + 3), axis, font=fonts.small, fill=LABEL)
        x = MARGIN
        for panel_image in panels[:2]:
            scaled = Image.fromarray(panel_image).resize((thumb - 8, thumb - 8), Image.BILINEAR)
            image.paste(scaled, (x, y + 26))
            x += thumb
        draw.text((text_x, y + 30), "判据", font=fonts.small, fill=LABEL)
        for line_index, line in enumerate(parts[task_id]["criterion"]):
            draw.text(
                (text_x + 46, y + 30 + line_index * 21), line, font=fonts.small, fill=VALUE
            )
        bottom = y + 30 + len(parts[task_id]["criterion"]) * 21 + 6
        draw.text(
            (text_x, bottom),
            f"自检 {parts[task_id]['status']} · spec {spec.hash()[:8]}"
            f" · 指令 {len(spec.instructions)}+{len(spec.holdout)}",
            font=fonts.small,
            fill=OK if parts[task_id]["status"].endswith("全绿") else WARN,
        )
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", help="level to card (repeatable)")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0, help="layout seed shown on the cards")
    parser.add_argument("--water", default=WATER_RUNG, choices=LADDER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    tasks = args.task or LEVEL_ORDER
    fonts = Fonts.load()
    status = selftest_rows()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    overview: list[tuple[str, list[np.ndarray], str]] = []
    for index, task_id in enumerate(tasks, 1):
        spec = get(task_id)
        env = TaskEnv(task_id, image_size=(args.size, args.size), max_episode_steps=10_000)
        try:
            env.reset(seed=args.seed)
            panels = level_panels(env, size=args.size, water_rung=args.water)
            level_status = status.get(task_id, "未运行")
            rows = characteristics(env, spec, level_status)
            card = build_card(
                task_id,
                panels,
                rows,
                fonts,
                footer=footer_for(task_id, env, args.water, level_status),
            )
        finally:
            env.close()
        path = args.out_dir / f"{index}_{task_id}.png"
        card.save(path)
        print(f"{path}  {card.size[0]}x{card.size[1]}  自检={status.get(task_id, '未运行')}")
        overview.append((task_id, panels, spec.difficulty_axis))

    sheet = build_overview(overview, size=args.size, status=status)
    sheet_path = args.out_dir / "overview.png"
    sheet.save(sheet_path)
    print(f"{sheet_path}  {sheet.size[0]}x{sheet.size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())