import scipy.io.wavfile
import scipy.signal


print('bemeneti filename:')
filename = input()


print('kimeneti filename:')
kimeneti_filename = input()

fs, wave = scipy.io.wavfile.read(filename)



with open(kimeneti_filename, "a") as a_file:
    for x in wave:
        a_file.write(str(x) + "\n")
print("\nkiírás befejeződött")