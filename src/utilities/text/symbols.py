import src.utilities.text.cmudict as cmudict

_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = "abcdefghijklmnopqrstuvwxyz"
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + _arpabet

# Special symbol ids
SPACE_ID = symbols.index(" ")
