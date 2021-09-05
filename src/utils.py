def type_conversion(data):
    """
    converts and returns the individual values of a pandas Series into either float type or None.   
    """
    try:
        return float(data)
    except ValueError:
        return None