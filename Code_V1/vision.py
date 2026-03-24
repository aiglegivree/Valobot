def detect_t_on_connected_mask(
    connected_mask: np.ndarray,
    cx_bot: int | None,
    width_bot: int,
    y_bot: int,
) -> tuple[bool, int | None]:
    """
    T detection tuned to be practical:
    - start from bottom stem
    - scan upward around bottom x
    - require strong width increase
    - require left and right extension
    - require stem continuity below
    - only light anti-turn filtering
    """
    if cx_bot is None or width_bot <= 0:
        return False, None

    h, w = connected_mask.shape
    cx_ref = w // 2

    # Keep this, but not too strict
    if abs(cx_bot - cx_ref) > T_CENTER_MAX_ERR:
        return False, None

    y_start = int(T_SCAN_Y_START_FRAC * h)
    y_end = int(T_SCAN_Y_END_FRAC * h)
    if y_start > y_end:
        y_start, y_end = y_end, y_start

    consecutive_hits = 0
    first_hit_row = None
    candidate_row = None
    candidate_x = None

    # small horizontal drift allowed while scanning upward
    last_x = cx_bot

    for y in range(y_end, y_start - 1, -1):
        row = connected_mask[y, :] > 0
        if not np.any(row):
            consecutive_hits = 0
            first_hit_row = None
            candidate_row = None
            candidate_x = None
            continue

        # search near previous x, not only exactly at cx_bot
        x_min, x_max = get_row_run_around_x(row, last_x)
        if x_min is None or x_max is None:
            consecutive_hits = 0
            first_hit_row = None
            candidate_row = None
            candidate_x = None
            continue

        run_cx = (x_min + x_max) // 2
        width = x_max - x_min

        # allow stem to move a bit
        stem_x = last_x
        stem_x = max(x_min, min(x_max, stem_x))

        left_ext = stem_x - x_min
        right_ext = x_max - stem_x

        width_ok = (
            width >= T_MIN_ABS_WIDTH and
            width >= T_WIDTH_RATIO * max(1, width_bot) and
            (width - width_bot) >= T_MIN_WIDTH_GAIN
        )

        both_sides_present = (
            left_ext >= T_MIN_SIDE_EXT and
            right_ext >= T_MIN_SIDE_EXT
        )

        # lighter balance filter than before
        smaller = min(left_ext, right_ext)
        larger = max(left_ext, right_ext)
        balance_ratio = smaller / max(1, larger)

        balanced_enough = balance_ratio >= 0.18

        bar_ok = width_ok and both_sides_present and balanced_enough

        if bar_ok:
            consecutive_hits += 1
            if first_hit_row is None:
                first_hit_row = y
            candidate_row = y
            candidate_x = run_cx
        else:
            consecutive_hits = 0
            first_hit_row = None
            candidate_row = None
            candidate_x = None

        # update tracking even before T hit
        last_x = run_cx

        if consecutive_hits >= T_MIN_BAR_ROWS:
            break

    if candidate_row is None or candidate_x is None:
        return False, None

    # validate stem continuity below bar
    y0 = candidate_row
    y1 = min(h - 1, candidate_row + T_STEM_MIN_ROWS)

    stem_hits = 0
    total_rows = max(1, y1 - y0 + 1)
    last_x = candidate_x

    for y in range(y0, y1 + 1):
        row = connected_mask[y, :] > 0
        x_min, x_max = get_row_run_around_x(row, last_x)
        if x_min is None:
            continue

        run_cx = (x_min + x_max) // 2
        last_x = run_cx

        x_left = max(0, run_cx - T_STEM_X_TOL)
        x_right = min(w, run_cx + T_STEM_X_TOL + 1)

        if np.any(connected_mask[y, x_left:x_right] > 0):
            stem_hits += 1

    required_hits = max(4, int(0.5 * total_rows))
    if stem_hits < required_hits:
        return False, None

    return True, first_hit_row