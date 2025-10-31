"""
Loto Skraceni Sistemi 
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""

# Nikolai formula
# predict the next point of intervals curve in 3 - 7 % of all cases
# average of all intervals is around 5.571428571428571 (39/7)   abs

from __future__ import print_function, division
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

from numpy import cos, sin, pi, absolute, arange


# Coefficient of Multiple Determination (R^2)
# Multiple Coefficient of Determination
M = 0.987444305500000


# parametters

A1 = 7.299684015518730
A2 = 0
A3 = -16.685835427847000
A4 = 4.033620068208560
A5 = 11.787890199614100
A6 = -0.929875140722455

B1 = -2.183088127022560
B2 = 5.622345763015300
B3 = -2.937935197869250
B4 = 3.055160052310020
B5 = 4.155657481996250
B6 = 1.999149991032180
B7 = 1.424034844968820
B8 = 1.123154320679130
B9 = 0.587528422335690
B10 = 2.192576277398270
B11 = 0.646184039028009
B12 = 3.128750813623030
B13 = -2.639908199110780
B14 = -1.234452659544030
B15 = 0
B16 = 0
B17 = 1.684322379296770
B18 = 1.806815397870690
B19 = -0.010080723947845
B20 = 1.819868387007100
B21 = 3.421929858053530
B22 = 1.862697127062110
B23 = 0.540543148349822
B24 = 1.862484902239440
B25 = 1.229198270286820
B26 = 1.888114102763830
B27 = 0.542228454843728
B28 = 1.853126551719710

C1 = 9.582199721922110
C2 = 160.540667471316000
C3 = 235.597570526473000
C4 = 193.064941311939000
C5 = -69.904752286696000
C6 = -85.770268955927000
C7 = 276.721054209067000
C8 = 0 
C9 = -374.987916855954000
C10 = 444.826536028089000
C11 = -256.601030945780000
C12 = -590.091497107117000
C13 = 173.815562399882000
C14 = -605.344333544192000

D1 = 100.982005590423000
D2 = -173.685727106493000
D3 = -28.816489276900800
D4 = -141.058426244729000
D5 = -172.520435212672000
D6 = -61.540771042905300
D7 = 114.003339618542000
D8 = 0 
D9 = -58.866476964092400
D10 = -51.716911912693900
D11 = -59.068870808688700
D12 = -55.721714142108400
D13 = -51.593058043094400
D14 = -64.539817955903400

E1 = 233.179121249626000
E2 = -72.647812742405900
E3 = 154.397943691535000
E4 = 66.306079684944700
E5 = 130.975552547632000
E6 = 114.372274193839000
E7 = -194.072161107444000
E8 = 16.274381953945800
E9 = 25.157528943044000
E10 = 35.300757969358900
E11 = 4.034054329422520
E12 = -72.271746132602100
E13 = 144.393758949961000
E14 = 17.511079644164100
E15 = 0
E16 = 0
E17 = -108.758489911582000
E18 = -69.399029881088400
E19 = -114.061251203356000
E20 = -70.397643655260600
E21 = -206.192826567812000
E22 = -71.161467289090500
E23 = -169.344108721358000
E24 = -72.593728034594300
E25 = -205.741600763855000
E26 = -73.906381152311700
E27 = -169.781637338030000
E28 = -67.941409230581000



t = pd.read_csv("/data/loto7_4502_k85.csv", header=None)


print()
print("###############################")
print()
print()
print('izvlacenja zadnjih 5 t')
print(t.tail(5).reset_index(drop=True))
print()
"""
###############################


izvlacenja zadnjih 5 t
   0   1   2   3   4   5   6
0  4  13  14  19  27  35  37
1  1   7  13  18  25  30  34
2  1   5   6   7  11  24  37
3  2   4   6  11  21  33  35
4  1   3  11  12  19  35  38
"""
print()
print()



####################################



# Minimalni i maksimalni dozvoljeni brojevi po poziciji
min_val = [1, 2, 3, 4, 5, 6, 7]
max_val = [33, 34, 35, 36, 37, 38, 39]

# Funkcija za mapiranje brojeva u indeksirani opseg [0..range_size-1]
def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        # Provera da li su svi brojevi u validnom opsegu
        if not df_indexed[i].between(0, max_val[i] - min_val[i]).all():
            raise ValueError(f"Vrednosti u koloni {i} nisu u opsegu 0 do {max_val[i] - min_val[i]}")
    return df_indexed

# Primena mapiranja
t_ix = map_to_indexed_range(t, min_val, max_val)


print()
print("###############################")
print()
print()
print('izvlacenja zadnjih 5 t_ix map')
print(t_ix.tail(5).reset_index(drop=True))
print()
"""
###############################


izvlacenja zadnjih 5 t_ix map
   0   1   2   3   4   5   6
0  3  11  11  15  22  29  30
1  0   5  10  14  20  24  27
2  0   3   3   3   6  18  30
3  1   2   3   7  16  27  28
4  0   1   8   8  14  29  31
"""
print()
print()



####################################

# prvih 7 kolona sadrze loto brojeve 
t_ix = t_ix.iloc[:, :7]

####################################




# suitable equation to reproduce all intervals curves for the 39 numbers

nikolai = A1 + A3 * sin(A4 + 
          C1 * cos(B1 * t + E1)  +  D1 * sin(B2 * t + E2) + 
          C2 * cos(B3 * t + E3)  +  D2 * sin(B4 * t + E4) + 
          C3 * cos(B5 * t + E5)  +  D3 * sin(B6 * t + E6) + 
          C4 * cos(B7 * t + E7)  +  D4 * sin(B8 * t + E8) + 
          C5 * cos(B9 * t + E9)  +  D5 * sin(B10 * t + E10) + 
          C6 * cos(B11 * t + E11) + D6 * sin(B12 * t + E12) + 
          C7 * cos(B13 * t + E13) + D7 * sin(B14 * t + E14)) + A5 * cos(A6 + 
          C9 * cos(B17 * t + E17)  +  D9 * sin(B18 * t + E18) +  
          C10 * cos(B19 * t + E19)  +  D10 * sin(B20 * t + E20) +  
          C11 * cos(B21 * t + E21)  +  D11 * sin(B22 * t + E22) +  
          C12 * cos(B23 * t + E23)  +  D12 * sin(B24 * t + E24) +  
          C13 * cos(B25 * t + E25)  +  D13 * sin(B26 * t + E26) + 
          C14 * cos(B27 * t + E27)  +  D14 * sin(B28 * t + E28))


print('nikolai zadnjih 5')
print(np.round(nikolai.tail(5).reset_index(drop=True)).astype(int))
print()
"""
nikolai zadnjih 5
    0   1   2   3   4   5   6
0  15 -19  11   6  16  10  10
1  34 -21 -19  31   9  28  -3
2  34  -4   5 -21  15   0  10
3  -8  15   5  15  29  30  10
4  34  -3  15  10   6  10  25
"""
print()
print()



nikolaiM = nikolai * M
print('nikolaiM zadnjih 5')
print(np.round(nikolaiM.tail(5).reset_index(drop=True)).astype(int))
print()
"""
nikolaiM zadnjih 5
    0   1   2   3   4   5   6
0  15 -19  11   6  16   9  10
1  33 -20 -19  31   9  28  -3
2  33  -4   5 -20  15   0  10
3  -8  15   5  15  28  29   9
4  33  -3  15  10   6   9  25
"""
print()
print()




nikolaiR = np.round(nikolai, 0)
print('nikolaiR zadnjih 5')
pprint(np.round(nikolaiR.tail(5).reset_index(drop=True)).astype(int))
print()
"""
nikolaiR zadnjih 5
    0   1   2   3   4   5   6
0  15 -19  11   6  16  10  10
1  34 -21 -19  31   9  28  -3
2  34  -4   5 -21  15   0  10
3  -8  15   5  15  29  30  10
4  34  -3  15  10   6  10  25
"""
print()
print()




nikolaiMR = np.round(nikolaiM, 0)
print('nikolaiMR zadnjih 5')
pprint(np.round(nikolaiMR.tail(5).reset_index(drop=True)).astype(int))
print()
"""
nikolaiMR zadnjih 5
    0   1   2   3   4   5   6
0  15 -19  11   6  16   9  10
1  33 -20 -19  31   9  28  -3
2  33  -4   5 -20  15   0  10
3  -8  15   5  15  28  29   9
4  33  -3  15  10   6   9  25
"""
print()
print()



print()
print('nikolai (i, f, fM)')
print()

for i in range(1,40,1):
          
          f = A1 + A3 * sin(A4 + 
          C1 * cos(B1 * i + E1)  +  D1 * sin(B2 * i + E2) + 
          C2 * cos(B3 * i + E3)  +  D2 * sin(B4 * i + E4) + 
          C3 * cos(B5 * i + E5)  +  D3 * sin(B6 * i + E6) + 
          C4 * cos(B7 * i + E7)  +  D4 * sin(B8 * i + E8) + 
          C5 * cos(B9 * i + E9)  +  D5 * sin(B10 * i + E10) + 
          C6 * cos(B11 * i + E11) + D6 * sin(B12 * i + E12) + 
          C7 * cos(B13 * i + E13) + D7 * sin(B14 * i + E14)) + A5 * cos(A6 + 
          C9 * cos(B17 * i + E17)  +  D9 * sin(B18 * i + E18) +  
          C10 * cos(B19 * i + E19)  +  D10 * sin(B20 * i + E20) +  
          C11 * cos(B21 * i + E21)  +  D11 * sin(B22 * i + E22) +  
          C12 * cos(B23 * i + E23)  +  D12 * sin(B24 * i + E24) +  
          C13 * cos(B25 * i + E25)  +  D13 * sin(B26 * i + E26) + 
          C14 * cos(B27 * i + E27)  +  D14 * sin(B28 * i + E28))

          fM = f * M
          
          print(np.round((i,f,fM), 0))
print() 
print()
"""
nikolai (i, f, fM)

[ 1. 34. 33.]
[ 2. -8. -8.]
[ 3. -3. -3.]
[ 4. 15. 15.]
[ 5. -4. -4.]
[6. 5. 5.]
[  7. -21. -20.]
[  8. -21. -21.]
[ 9. 11. 11.]
[10. 17. 16.]
[11. 15. 15.]
[12. 10. 10.]
[ 13. -19. -19.]
[14. 11. 11.]
[15. -8. -8.]
[16. 15. 14.]
[17. 35. 34.]
[18. 31. 31.]
[19.  6.  6.]
[20. 22. 22.]
[21. 29. 28.]
[22. 14. 14.]
[23.  9.  9.]
[24. -0. -0.]
[25.  9.  9.]
[ 26. -20. -20.]
[27. 16. 16.]
[ 28. -10. -10.]
[29.  3.  3.]
[30. 28. 28.]
[31.  5.  5.]
[32. -5. -5.]
[33. 30. 29.]
[34. -3. -3.]
[35. 10.  9.]
[36.  3.  3.]
[37. 10. 10.]
[38. 25. 25.]
[39. 16. 16.]
"""
print()
print()
print()



z = np.linspace(-10,10,100)
sigmoid =1/(1+np.exp(-z))
plt.plot(z, sigmoid,'k')
plt.grid()
plt.show()



print('Hermitian t_ix zadnjih 5 map')
print('Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum')
print(np.round(np.fft.hfft(t_ix.tail(5).reset_index(drop=True)), 0))
print()
"""
Hermitian t_ix zadnjih 5 map
Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum
[[209. -69.  10.  -5. -10.  -7. -11.  -7. -10.  -5.  10. -69.]
 [173. -70.  -2.  -7.  -4.  -4.   1.  -4.  -4.  -7.  -2. -70.]
 [ 96. -59.  36. -24.   6.  -7.   0.  -7.   6. -24.  36. -59.]
 [139. -83.  25.  -1.  -5.   3.  -5.   3.  -5.  -1.  25. -83.]
 [151. -85.  23. -19.  -5.  11.  -1.  11.  -5. -19.  23. -85.]]
"""
print()
print()



print('Hermitian nikolai zadnjih 5')
print('Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum')
print(np.round(np.fft.hfft(nikolai.tail(5).reset_index(drop=True)), 0))
print()
"""
Hermitian nikolai zadnjih 5
Compute the FFT of a signal that has Hermitian symmetry, i.e., a real spectrum
[[ 73. -51. -25.  15.  21.  49.  87.  49.  21.  15. -25. -51.]
 [ 88. -77. -14.  95.  96.  93. -67.  93.  96.  95. -14. -77.]
 [ 34.   6.  61.  43. -13.  21. 134.  21. -13.  43.  61.   6.]
 [187. -67. -17.  29. -47. -15. -50. -15. -47.  29. -17. -67.]
 [135.  -5.  24.  -8.  52.  39.  68.  39.  52.  -8.  24.  -5.]]
"""
print()
print()



print()
print('Hermitian i t_ix zadnjih 5 map')
print('Compute the inverse FFT of a signal that has Hermitian symmetry')
print(np.round(np.fft.ihfft(t_ix.tail(5).reset_index(drop=True)), 0))
print()
"""
Hermitian i t_ix zadnjih 5 map
Compute the inverse FFT of a signal that has Hermitian symmetry
[[17.-0.j -2.-5.j -3.-1.j -2.-0.j]
 [14.-0.j -3.-5.j -2.-2.j -2.-1.j]
 [ 9.-0.j  1.-5.j -3.-2.j -3.-0.j]
 [12.-0.j -1.-7.j -3.-1.j -2.-0.j]
 [13.-0.j -1.-7.j -4.-2.j -2.-0.j]]
"""
print()
print()



print('Hermitian i nikolai zadnjih 5')
print('Compute the inverse FFT of a signal that has Hermitian symmetry')
print(np.round(np.fft.ihfft(nikolai.tail(5).reset_index(drop=True)), 0))
print()
"""
Hermitian i nikolai zadnjih 5
Compute the inverse FFT of a signal that has Hermitian symmetry
[[ 7.-0.j -2.-4.j  2.-3.j  4.-3.j]
 [ 8.-0.j -3.-7.j  8.-2.j  7.+7.j]
 [ 6.-0.j  6.-3.j  3.+2.j  5.-6.j]
 [13.-0.j -6.-4.j -3.+4.j -3.+1.j]
 [14.-0.j  4.-2.j  2.-5.j  4.-2.j]]
"""
print()
print()



print()
print('Helper routins t_x zadnjih 5 map')
print('Shift the zero-frequency component to the center of the spectrum')
print(np.round(np.fft.fftshift(t_ix.tail(5).reset_index(drop=True)), 0))
print()
"""
Helper routins t_x zadnjih 5 map
Shift the zero-frequency component to the center of the spectrum
[[16 27 28  1  2  3  7]
 [14 29 31  0  1  8  8]
 [22 29 30  3 11 11 15]
 [20 24 27  0  5 10 14]
 [ 6 18 30  0  3  3  3]]
"""
print()
print()



print('Helper routins nikolai zadnjih 5')
print('Shift the zero-frequency component to the center of the spectrum')
print(np.round(np.fft.fftshift(nikolai.tail(5).reset_index(drop=True)), 0))
print()
"""
Helper routins nikolai zadnjih 5
Shift the zero-frequency component to the center of the spectrum
[[ 29.  30.  10.  -8.  15.   5.  15.]
 [  6.  10.  25.  34.  -3.  15.  10.]
 [ 16.  10.  10.  15. -19.  11.   6.]
 [  9.  28.  -3.  34. -21. -19.  31.]
 [ 15.  -0.  10.  34.  -4.   5. -21.]]
"""
print()
print()



print()
print('Helper routins t_ix zadnjih 5 map')
print('The inverse of fftshift')
print(np.round(np.fft.ifftshift(t_ix.tail(5).reset_index(drop=True)), 0))
print()
"""
Helper routins t_ix zadnjih 5 map
The inverse of fftshift
[[ 3  6 18 30  0  3  3]
 [ 7 16 27 28  1  2  3]
 [ 8 14 29 31  0  1  8]
 [15 22 29 30  3 11 11]
 [14 20 24 27  0  5 10]]
"""
print()
print()



print('Helper routins nikolai zadnjih 5')
print('The inverse of fftshift')
print(np.round(np.fft.ifftshift(nikolai.tail(5).reset_index(drop=True)), 0))
print()
"""
Helper routins nikolai zadnjih 5
The inverse of fftshift
[[-21.  15.  -0.  10.  34.  -4.   5.]
 [ 15.  29.  30.  10.  -8.  15.   5.]
 [ 10.   6.  10.  25.  34.  -3.  15.]
 [  6.  16.  10.  10.  15. -19.  11.]
 [ 31.   9.  28.  -3.  34. -21. -19.]]
"""
print()
print()



#################################################



print()
print("###############################")
print()

print('Helper routins t_ix map')
print('Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft)')
print(np.round(np.fft.rfft(t_ix.tail(5).reset_index(drop=True)), 0))
print()
"""
###############################

Helper routins t_ix map
Return the Discrete Fourier Transform sample frequencies (for usage with rfft, irfft)
[[121. +0.j -14.+35.j -19. +5.j -17. +1.j]
 [100. +0.j -18.+33.j -17.+11.j -15. +4.j]
 [ 63. +0.j   8.+37.j -21.+17.j -19. +3.j]
 [ 84. +0.j  -8.+48.j -18. +8.j -12. +1.j]
 [ 91. +0.j  -8.+47.j -27.+15.j -11. +2.j]]
"""
print()
print()



print()
print('Helper routins nikolai')
print('Return the Discrete Fourier Transform sample frequencies')
print(np.round(np.fft.rfft(nikolai.tail(5).reset_index(drop=True)), 0))
print()
"""
Helper routins nikolai
Return the Discrete Fourier Transform sample frequencies
[[ 49. +0.j -16.+26.j  12.+22.j  31.+24.j]
 [ 59. +0.j -20.+51.j  56.+14.j  52.-51.j]
 [ 39. +0.j  42.+22.j  24.-11.j  33.+45.j]
 [ 94. +0.j -40.+26.j -18.-27.j -18. -8.j]
 [ 97. +0.j  27.+15.j  17.+33.j  25.+13.j]]
"""
print()
print()



print()
print('Helper routins i t_ix')
print('Return the Discrete Inverse Fourier Transform sample frequencies')
print(np.round(np.fft.irfft(t_ix.tail(5).reset_index(drop=True)), 0))
print()
"""
Helper routins i t_ix
Return the Discrete Inverse Fourier Transform sample frequencies
[[17. -6.  1. -0. -1. -1. -1. -1. -1. -0.  1. -6.]
 [14. -6. -0. -1. -0. -0.  0. -0. -0. -1. -0. -6.]
 [ 8. -5.  3. -2.  0. -1.  0. -1.  0. -2.  3. -5.]
 [12. -7.  2. -0. -0.  0. -0.  0. -0. -0.  2. -7.]
 [13. -7.  2. -2. -0.  1. -0.  1. -0. -2.  2. -7.]]
"""
print()
print()



print()
print('Helper routins i nikolai')
print('Return the Discrete Inverse Fourier Transform sample frequencies')
print(np.round(np.fft.irfft(nikolai.tail(5).reset_index(drop=True)), 0))
print()
""""
Helper routins i nikolai
Return the Discrete Inverse Fourier Transform sample frequencies
[[ 6. -4. -2.  1.  2.  4.  7.  4.  2.  1. -2. -4.]
 [ 7. -6. -1.  8.  8.  8. -6.  8.  8.  8. -1. -6.]
 [ 3.  1.  5.  4. -1.  2. 11.  2. -1.  4.  5.  1.]
 [16. -6. -1.  2. -4. -1. -4. -1. -4.  2. -1. -6.]
 [11. -0.  2. -1.  4.  3.  6.  3.  4. -1.  2. -0.]]
"""
print()
print()



###############################


print()
print()
print("###############################")
print()



################################################
