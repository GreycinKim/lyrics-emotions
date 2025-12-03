def fix_mojibake(s: str) -> str:
    """
    Fix strings like 'AxÃ©' -> 'Axé' when mis-decoded.
    If it's already fine, returns original text.
    """
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
