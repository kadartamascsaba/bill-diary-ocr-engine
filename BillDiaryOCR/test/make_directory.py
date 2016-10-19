import os

alphabet = "abcdefghijklmnopqrstuvwxyz"
# alphabet = alphabet.upper()
# alphabet = "0123456789"
# alphabet = "-,.%!:'/*()="

for s in alphabet:
	if not os.path.exists("small_" + s):
		os.makedirs("small_" + s)
