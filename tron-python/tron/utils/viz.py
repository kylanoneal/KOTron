import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tron.ai.benchmarks import Tactic, TacticResult
from tron.game import GameStatus, PovGameState, get_status


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class BoardVizOptions:
    cell_px: int = 36
    grid_px: int = 2
    margin_px: int = 8
    background_color: Color = (17, 24, 39)
    empty_color: Color = (31, 41, 55)
    wall_color: Color = (75, 85, 99)
    hero_color: Color = (37, 99, 235)
    opponent_color: Color = (220, 38, 38)
    collision_color: Color = (124, 58, 237)
    grid_color: Color = (17, 24, 39)
    player_label_color: Color = (249, 250, 251)
    inactive_player_alpha: float = 0.45
    show_player_labels: bool = True
    show_outcome: bool = True
    hero_win_color: Color = (37, 99, 235)
    opponent_win_color: Color = (220, 38, 38)
    tie_color: Color = (156, 163, 175)
    outcome_overlay_alpha: float = 0.22
    outcome_outline_px: int = 4
    font_path: Optional[str] = None
    player_label_font_px: int = 14


@dataclass(frozen=True)
class TacticRowVizOptions:
    board: BoardVizOptions = field(default_factory=BoardVizOptions)
    background_color: Color = (17, 24, 39)
    text_color: Color = (209, 213, 219)
    pass_text_color: Color = (34, 197, 94)
    fail_text_color: Color = (239, 68, 68)
    score_text_color: Color = (249, 250, 251)
    pass_row_tint_color: Color = (34, 197, 94)
    fail_row_tint_color: Color = (239, 68, 68)
    row_tint_alpha: float = 0.08
    placeholder_text_color: Color = (156, 163, 175)
    arrow_color: Color = (209, 213, 219)
    row_padding_px: int = 14
    status_column_width_px: int = 128
    status_column_gap_px: int = 20
    board_gap_px: int = 64
    board_text_gap_px: int = 10
    text_box_height_px: int = 72
    text_padding_px: int = 6
    text_line_spacing_px: int = 4
    text_font_px: int = 13
    status_font_px: int = 40
    score_format: str = "Score: {score:.2f}"
    arrow_thickness_px: int = 3
    arrow_head_px: int = 12
    font_path: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkVizOptions:
    row: TacticRowVizOptions = field(default_factory=TacticRowVizOptions)
    background_color: Color = (17, 24, 39)
    title_color: Color = (249, 250, 251)
    separator_color: Color = (75, 85, 99)
    outer_padding_px: int = 18
    title_height_px: int = 64
    title_font_px: int = 64
    row_gap_px: int = 22
    separator_px: int = 2
    font_path: Optional[str] = None


def render_game_state_image(
    pov_game_state: PovGameState,
    options: Optional[BoardVizOptions] = None,
) -> np.ndarray:
    """Render one POV game state as an RGB image.

    Args:
        pov_game_state: Game state plus hero/opponent player indices.
        options: Rendering options for board geometry and colors.

    Returns:
        An RGB numpy image with shape ``(height, width, 3)``.
    """
    options = options or BoardVizOptions()
    game_state = pov_game_state.game_state

    grid_width = game_state.num_cols * options.cell_px + (game_state.num_cols + 1) * options.grid_px
    grid_height = game_state.num_rows * options.cell_px + (game_state.num_rows + 1) * options.grid_px
    width = grid_width + 2 * options.margin_px
    height = grid_height + 2 * options.margin_px

    image = _make_canvas(width, height, options.background_color)

    gx = options.margin_px
    gy = options.margin_px
    image[gy : gy + grid_height, gx : gx + grid_width] = options.grid_color

    for row in range(game_state.num_rows):
        for col in range(game_state.num_cols):
            idx = row * game_state.num_cols + col
            color = options.wall_color if game_state.board & (1 << idx) else options.empty_color
            x0, y0 = _cell_origin(row, col, options)
            image[gy + y0 : gy + y0 + options.cell_px, gx + x0 : gx + x0 + options.cell_px] = color

    hero = game_state.players[pov_game_state.hero_index]
    opponent = game_state.players[pov_game_state.opponent_index]

    player_cells: Dict[int, List[Tuple[str, Color, bool]]] = {}
    player_cells.setdefault(hero.idx, []).append(("H", options.hero_color, hero.can_move))
    player_cells.setdefault(opponent.idx, []).append(("O", options.opponent_color, opponent.can_move))

    for idx, entries in player_cells.items():
        row = idx // game_state.num_cols
        col = idx % game_state.num_cols
        x0, y0 = _cell_origin(row, col, options)

        if len(entries) > 1:
            color = options.collision_color
        else:
            _, player_color, can_move = entries[0]
            alpha = 1.0 if can_move else options.inactive_player_alpha
            color = _blend(player_color, options.empty_color, alpha)

        image[gy + y0 : gy + y0 + options.cell_px, gx + x0 : gx + x0 + options.cell_px] = color

    outcome_color = _get_outcome_color(pov_game_state, options)
    if outcome_color is not None:
        image = _apply_outcome_style(
            image,
            box=(gx, gy, gx + grid_width, gy + grid_height),
            color=outcome_color,
            overlay_alpha=options.outcome_overlay_alpha,
            outline_px=options.outcome_outline_px,
        )

    if options.show_player_labels:
        font = _load_font(options.font_path, options.player_label_font_px)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for idx, entries in player_cells.items():
            row = idx // game_state.num_cols
            col = idx % game_state.num_cols
            x0, y0 = _cell_origin(row, col, options)
            label = "/".join(label for label, _, _ in entries)

            cx = gx + x0 + options.cell_px / 2
            cy = gy + y0 + options.cell_px / 2
            _draw_centered_text_pil(
                draw,
                label,
                (cx, cy),
                font,
                options.player_label_color,
            )

        image = np.array(pil_image, dtype=np.uint8)

    return image


def render_tactic_row_image(
    tactic: Tactic,
    result: TacticResult,
    options: Optional[TacticRowVizOptions] = None,
    *,
    score_passfail: bool = True,
) -> np.ndarray:
    """Render one tactic result as a horizontal row image.

    Args:
        tactic: Tactic containing expected hero and opponent moves.
        result: Actual states and hero moves produced by the tactic run.
        options: Rendering options for row spacing, arrows, and text.
        score_passfail: Whether to show pass/fail text instead of a numeric score.

    Returns:
        An RGB numpy image with shape ``(height, width, 3)``.
    """
    options = options or TacticRowVizOptions()

    if not result.pov_game_states:
        raise ValueError("TacticResult must contain at least one game state.")

    board_images = [
        render_game_state_image(pov_game_state, options.board)
        for pov_game_state in result.pov_game_states
    ]

    board_height, board_width, _ = board_images[0].shape
    slot_count = max(len(board_images), len(result.actual_hero_dirs) + 1)

    row_width = (
        2 * options.row_padding_px
        + options.status_column_width_px
        + options.status_column_gap_px
        + slot_count * board_width
        + (slot_count - 1) * options.board_gap_px
    )
    row_height = (
        2 * options.row_padding_px
        + board_height
        + options.board_text_gap_px
        + options.text_box_height_px
    )

    image = _make_canvas(row_width, row_height, options.background_color)
    text_font = _load_font(options.font_path, options.text_font_px)
    status_font = _load_font(options.font_path, options.status_font_px)

    board_y = options.row_padding_px
    text_y = board_y + board_height + options.board_text_gap_px
    board_start_x = (
        options.row_padding_px
        + options.status_column_width_px
        + options.status_column_gap_px
    )

    status_text, status_color = _tactic_status_text_and_color(
        tactic,
        result,
        score_passfail,
        options,
    )
    image = _draw_centered_text(
        image,
        status_text,
        (
            options.row_padding_px,
            options.row_padding_px,
            options.row_padding_px + options.status_column_width_px,
            row_height - options.row_padding_px,
        ),
        status_font,
        status_color,
    )

    for slot_idx in range(slot_count):
        x = board_start_x + slot_idx * (board_width + options.board_gap_px)

        if slot_idx < len(board_images):
            _paste(image, board_images[slot_idx], x, board_y)
        else:
            placeholder = _make_canvas(board_width, board_height, options.background_color)
            placeholder = _draw_centered_text(
                placeholder,
                "No state\nproduced",
                (0, 0, board_width, board_height),
                text_font,
                options.placeholder_text_color,
            )
            _paste(image, placeholder, x, board_y)

        if slot_idx < len(result.actual_hero_dirs):
            text_block = _draw_text_block(
                width_px=board_width,
                height_px=options.text_box_height_px,
                lines=_move_lines(tactic, result, slot_idx),
                background_color=options.background_color,
                text_color=options.text_color,
                font=text_font,
                padding_px=options.text_padding_px,
                line_spacing_px=options.text_line_spacing_px,
            )
            _paste(image, text_block, x, text_y)

    for slot_idx in range(len(board_images) - 1):
        start_x = board_start_x + slot_idx * (board_width + options.board_gap_px) + board_width + 8
        end_x = board_start_x + (slot_idx + 1) * (board_width + options.board_gap_px) - 8
        y = board_y + board_height // 2

        image = _draw_arrow(
            image,
            start=(start_x, y),
            end=(end_x, y),
            color=options.arrow_color,
            thickness_px=options.arrow_thickness_px,
            head_px=options.arrow_head_px,
        )

    if score_passfail:
        tint_color = (
            options.pass_row_tint_color
            if _tactic_passed(tactic, result)
            else options.fail_row_tint_color
        )
        image = _apply_tint(image, tint_color, options.row_tint_alpha)

    return image


def render_tactic_benchmark_image(
    results: Sequence[TacticResult],
    score_passfail: bool = True,
    options: Optional[BenchmarkVizOptions] = None,
    max_rows: Optional[int] = None,
) -> np.ndarray:
    """Render the full tactic benchmark output as one RGB image.

    Args:

        results: Tactic results returned by ``run_tactic``.
        options: Rendering options for title, spacing, and separators.
        max_rows: Optional limit for the number of rows rendered.

    Returns:
        An RGB numpy image with shape ``(height, width, 3)``.
    """
    options = options or BenchmarkVizOptions()

    tactics = [r.tactic for r in results]

    pairs = list(zip(tactics, results))
    if max_rows is not None:
        pairs = pairs[:max_rows]

    if not pairs:
        raise ValueError("At least one tactic/result pair is required.")

    row_images = [
        render_tactic_row_image(
            tactic,
            result,
            options.row,
            score_passfail=score_passfail,
        )
        for tactic, result in pairs
    ]

    content_width = max(row.shape[1] for row in row_images)
    padded_rows = [
        _pad_to_width(row, content_width, options.background_color)
        for row in row_images
    ]

    width = content_width + 2 * options.outer_padding_px
    height = (
        2 * options.outer_padding_px
        + options.title_height_px
        + sum(row.shape[0] for row in padded_rows)
        + (len(padded_rows) - 1) * (options.row_gap_px + options.separator_px)
    )

    image = _make_canvas(width, height, options.background_color)

    title_font = _load_font(options.font_path, options.title_font_px)

    if score_passfail:

        passed = 0

        for r in results:
            if r.correct_moves == len(r.tactic.opposing_dirs):
                passed += 1

        title = f"Passed {passed} / {len(results)}"

    else:

        score = 0

        for r in results:
            score += r.correct_moves / len(r.tactic.opposing_dirs)


        title = f"Average score: {score / len(results)}"


    image = _draw_centered_text(
        image,
        title,
        (
            options.outer_padding_px,
            options.outer_padding_px,
            width - options.outer_padding_px,
            options.outer_padding_px + options.title_height_px,
        ),
        title_font,
        options.title_color,
    )

    y = options.outer_padding_px + options.title_height_px

    for row_idx, row in enumerate(padded_rows):
        _paste(image, row, options.outer_padding_px, y)
        y += row.shape[0]

        if row_idx < len(padded_rows) - 1:
            y += options.row_gap_px // 2
            image[
                y : y + options.separator_px,
                options.outer_padding_px : options.outer_padding_px + content_width,
            ] = options.separator_color
            y += options.separator_px + options.row_gap_px - options.row_gap_px // 2

    return image


def _move_lines(tactic: Tactic, result: TacticResult, step: int) -> List[str]:
    """Build display lines for one tactic step.

    Args:
        tactic: Tactic containing expected and opponent moves.
        result: Tactic result containing actual hero moves.
        step: Move index to describe.

    Returns:
        Lines to render under a game state.
    """
    if tactic.expected_hero_dirs is None:
        expected = "ANY"
    elif step < len(tactic.expected_hero_dirs):
        expected = _direction_name(tactic.expected_hero_dirs[step])
    else:
        expected = "-"

    actual = (
        _direction_name(result.actual_hero_dirs[step])
        if step < len(result.actual_hero_dirs)
        else "-"
    )
    opponent = (
        _direction_name(tactic.opposing_dirs[step])
        if step < len(tactic.opposing_dirs)
        else "-"
    )

    return [
        f"Expected hero move: {expected}",
        f"Actual hero move: {actual}",
        f"Opponent move: {opponent}",
    ]


def _direction_name(direction: object) -> str:
    """Return the display name for a direction-like value."""
    return getattr(direction, "name", str(direction))


def _tactic_status_text_and_color(
    tactic: Tactic,
    result: TacticResult,
    score_passfail: bool,
    options: TacticRowVizOptions,
) -> Tuple[str, Color]:
    """Return the left-column tactic status label and color."""
    if score_passfail:
        if _tactic_passed(tactic, result):
            return "Pass", options.pass_text_color

        return "Fail", options.fail_text_color

    return (
        options.score_format.format(score=_tactic_score(tactic, result)),
        options.score_text_color,
    )


def _tactic_score(tactic: Tactic, result: TacticResult) -> float:
    """Return the fraction of opposing moves answered correctly."""
    return result.correct_moves / len(tactic.opposing_dirs)


def _tactic_passed(tactic: Tactic, result: TacticResult) -> bool:
    """Return whether every opposing move was answered correctly."""
    return result.correct_moves == len(tactic.opposing_dirs)


def _get_outcome_color(
    pov_game_state: PovGameState,
    options: BoardVizOptions,
) -> Optional[Color]:
    """Return the POV outcome color for a completed game state."""
    if not options.show_outcome:
        return None

    status_info = get_status(pov_game_state.game_state)
    if status_info.status == GameStatus.IN_PROGRESS:
        return None

    if status_info.status == GameStatus.TIE:
        return options.tie_color

    if status_info.winner_index == pov_game_state.hero_index:
        return options.hero_win_color

    if status_info.winner_index == pov_game_state.opponent_index:
        return options.opponent_win_color

    return None


def _cell_origin(row: int, col: int, options: BoardVizOptions) -> Tuple[int, int]:
    """Return the top-left pixel for a board cell within the grid."""
    x = options.grid_px + col * (options.cell_px + options.grid_px)
    y = options.grid_px + row * (options.cell_px + options.grid_px)
    return x, y


def _make_canvas(width_px: int, height_px: int, color: Color) -> np.ndarray:
    """Create an RGB canvas filled with one color."""
    image = np.empty((height_px, width_px, 3), dtype=np.uint8)
    image[:, :] = color
    return image


def _apply_outcome_style(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Color,
    overlay_alpha: float,
    outline_px: int,
) -> np.ndarray:
    """Apply a translucent overlay and outline to a rectangular region."""
    left, top, right, bottom = box
    styled = image.copy()

    overlay = np.array(color, dtype=np.float32)
    region = styled[top:bottom, left:right].astype(np.float32)
    region = region * (1.0 - overlay_alpha) + overlay * overlay_alpha
    styled[top:bottom, left:right] = np.clip(region, 0, 255).astype(np.uint8)

    if outline_px > 0:
        outline_color = np.array(color, dtype=np.uint8)
        styled[top : top + outline_px, left:right] = outline_color
        styled[bottom - outline_px : bottom, left:right] = outline_color
        styled[top:bottom, left : left + outline_px] = outline_color
        styled[top:bottom, right - outline_px : right] = outline_color

    return styled


def _paste(destination: np.ndarray, source: np.ndarray, x: int, y: int) -> None:
    """Paste one RGB image into another."""
    height, width, _ = source.shape
    destination[y : y + height, x : x + width] = source


def _pad_to_width(image: np.ndarray, width_px: int, color: Color, alignment="left") -> np.ndarray:
    """Left or center-pad an RGB image to a target width."""
    if image.shape[1] >= width_px:
        return image

    padded = _make_canvas(width_px, image.shape[0], color)

    if alignment=="center":
        x = (width_px - image.shape[1]) // 2
    elif alignment=="left":
        x = 0
    else:
        raise NotImplementedError()
    
    _paste(padded, image, x, 0)
    return padded


def _blend(foreground: Color, background: Color, alpha: float) -> Color:
    """Blend two RGB colors."""
    return tuple(
        int(round(foreground[i] * alpha + background[i] * (1.0 - alpha)))
        for i in range(3)
    )


def _apply_tint(image: np.ndarray, color: Color, alpha: float) -> np.ndarray:
    """Apply a translucent color tint to an RGB image."""
    tint = np.array(color, dtype=np.float32)
    tinted = image.astype(np.float32) * (1.0 - alpha) + tint * alpha
    return np.clip(tinted, 0, 255).astype(np.uint8)


def _load_font(font_path: Optional[str], size_px: int) -> ImageFont.ImageFont:
    """Load a TrueType font with a reasonable fallback."""
    if font_path is not None:
        try:
            return ImageFont.truetype(font_path, size_px)
        except OSError:
            pass

    for candidate in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size_px)
        except OSError:
            pass

    return ImageFont.load_default()


def _draw_text_block(
    width_px: int,
    height_px: int,
    lines: Sequence[str],
    background_color: Color,
    text_color: Color,
    font: ImageFont.ImageFont,
    padding_px: int,
    line_spacing_px: int,
) -> np.ndarray:
    """Draw left-aligned multiline text into a fixed-size RGB image."""
    image = _make_canvas(width_px, height_px, background_color)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    draw.multiline_text(
        (padding_px, padding_px),
        "\n".join(lines),
        fill=text_color,
        font=font,
        spacing=line_spacing_px,
    )
    return np.array(pil_image, dtype=np.uint8)


def _draw_centered_text(
    image: np.ndarray,
    text: str,
    box: Tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    color: Color,
) -> np.ndarray:
    """Draw centered multiline text into a rectangular region."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = left + (right - left - text_width) / 2 - bbox[0]
    y = top + (bottom - top - text_height) / 2 - bbox[1]

    draw.multiline_text(
        (x, y),
        text,
        fill=color,
        font=font,
        spacing=4,
        align="center",
    )

    return np.array(pil_image, dtype=np.uint8)


def _draw_centered_text_pil(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: Tuple[float, float],
    font: ImageFont.ImageFont,
    color: Color,
) -> None:
    """Draw centered single-line text using an existing PIL draw object."""
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = center[0] - text_width / 2 - bbox[0]
    y = center[1] - text_height / 2 - bbox[1]

    draw.text((x, y), text, fill=color, font=font)


def _draw_arrow(
    image: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Color,
    thickness_px: int,
    head_px: int,
) -> np.ndarray:
    """Draw a horizontal arrow onto an RGB image."""
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    draw.line([start, end], fill=color, width=thickness_px)

    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = angle + math.pi - math.pi / 6
    right = angle + math.pi + math.pi / 6

    arrow_head = [
        end,
        (
            int(end[0] + head_px * math.cos(left)),
            int(end[1] + head_px * math.sin(left)),
        ),
        (
            int(end[0] + head_px * math.cos(right)),
            int(end[1] + head_px * math.sin(right)),
        ),
    ]

    draw.polygon(arrow_head, fill=color)
    return np.array(pil_image, dtype=np.uint8)
