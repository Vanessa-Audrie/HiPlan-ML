def hours_to_hh_mm(hours: float) -> str:
    total_minutes = int(round(hours * 60))
    hh = total_minutes // 60
    mm = total_minutes % 60
    return f"{hh} jam {mm} menit"