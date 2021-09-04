def type_conversion(data):
    try:
        return float(data)
    except ValueError:
        return None