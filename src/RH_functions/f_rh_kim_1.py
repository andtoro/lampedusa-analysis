import pandas as pd

def func1(dt, rh, column):
    frh = pd.DataFrame({'time': dt})
    frh['f_S'] = 0
    frh['f_L'] = 0
    frh['f_SS'] = 0

    for i in range(len(rh[column])):
        if 0 <=rh[column][i]*100 <= 36:
            frh['f_S'][i] = 1.00
            frh['f_L'][i] = 1.00
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 37:
            frh['f_S'][i] = 1.38
            frh['f_L'][i] = 1.31
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 38:
            frh['f_S'][i] = 1.40
            frh['f_L'][i] = 1.32
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 39:
            frh['f_S'][i] = 1.42
            frh['f_L'][i] = 1.34
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 40:
            frh['f_S'][i] = 1.44
            frh['f_L'][i] = 1.35
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 41:
            frh['f_S'][i] = 1.46
            frh['f_L'][i] = 1.36
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 42:
            frh['f_S'][i] = 1.48
            frh['f_L'][i] = 1.38
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 43:
            frh['f_S'][i] = 1.49
            frh['f_L'][i] = 1.39
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 44:
            frh['f_S'][i] = 1.51
            frh['f_L'][i] = 1.41
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 45:
            frh['f_S'][i] = 1.53
            frh['f_L'][i] = 1.42
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 46:
            frh['f_S'][i] = 1.55
            frh['f_L'][i] = 1.44
            frh['f_SS'][i] = 1.00
        elif rh[column][i]*100 == 47:
            frh['f_S'][i] = 1.57
            frh['f_L'][i] = 1.45
            frh['f_SS'][i] = 2.3584
        elif rh[column][i]*100 == 48:
            frh['f_S'][i] = 1.59
            frh['f_L'][i] = 1.47
            frh['f_SS'][i] = 2.3799
        elif rh[column][i]*100 == 49:
            frh['f_S'][i] = 1.62
            frh['f_L'][i] = 1.49
            frh['f_SS'][i] = 2.4204
        elif rh[column][i]*100 == 50:
            frh['f_S'][i] = 1.64
            frh['f_L'][i] = 1.50
            frh['f_SS'][i] = 2.4488
        elif rh[column][i]*100 == 51:
            frh['f_S'][i] = 1.66
            frh['f_L'][i] = 1.52
            frh['f_SS'][i] = 2.4848
        elif rh[column][i]*100 == 52:
            frh['f_S'][i] = 1.68
            frh['f_L'][i] = 1.54
            frh['f_SS'][i] = 2.5006
        elif rh[column][i]*100 == 53:
            frh['f_S'][i] = 1.71
            frh['f_L'][i] = 1.55
            frh['f_SS'][i] = 2.5052
        elif rh[column][i]*100 == 54:
            frh['f_S'][i] = 1.73
            frh['f_L'][i] = 1.57
            frh['f_SS'][i] = 2.5279
        elif rh[column][i]*100 == 55:
            frh['f_S'][i] = 1.76
            frh['f_L'][i] = 1.59
            frh['f_SS'][i] = 2.5614
        elif rh[column][i]*100 == 56:
            frh['f_S'][i] = 1.78
            frh['f_L'][i] = 1.61
            frh['f_SS'][i] = 2.5848
        elif rh[column][i]*100 == 57:
            frh['f_S'][i] = 1.81
            frh['f_L'][i] = 1.63
            frh['f_SS'][i] = 2.5888
        elif rh[column][i]*100 == 58:
            frh['f_S'][i] = 1.83
            frh['f_L'][i] = 1.65
            frh['f_SS'][i] = 2.6160
        elif rh[column][i]*100 == 59:
            frh['f_S'][i] = 1.86
            frh['f_L'][i] = 1.67
            frh['f_SS'][i] = 2.6581
        elif rh[column][i]*100 == 60:
            frh['f_S'][i] = 1.89
            frh['f_L'][i] = 1.69
            frh['f_SS'][i] = 2.6866
        elif rh[column][i]*100 == 61:
            frh['f_S'][i] = 1.92
            frh['f_L'][i] = 1.71
            frh['f_SS'][i] = 2.7341
        elif rh[column][i]*100 == 62:
            frh['f_S'][i] = 1.95
            frh['f_L'][i] = 1.73
            frh['f_SS'][i] = 2.7834
        elif rh[column][i]*100 == 63:
            frh['f_S'][i] = 1.99
            frh['f_L'][i] = 1.75
            frh['f_SS'][i] = 2.8272
        elif rh[column][i]*100 == 64:
            frh['f_S'][i] = 2.02
            frh['f_L'][i] = 1.78
            frh['f_SS'][i] = 2.8287
        elif rh[column][i]*100 == 65:
            frh['f_S'][i] = 2.06
            frh['f_L'][i] = 1.80
            frh['f_SS'][i] = 2.8594
        elif rh[column][i]*100 == 66:
            frh['f_S'][i] = 2.09
            frh['f_L'][i] = 1.83
            frh['f_SS'][i] = 2.8943
        elif rh[column][i]*100 == 67:
            frh['f_S'][i] = 2.13
            frh['f_L'][i] = 1.86
            frh['f_SS'][i] = 2.9105
        elif rh[column][i]*100 == 68:
            frh['f_S'][i] = 2.17
            frh['f_L'][i] = 1.89
            frh['f_SS'][i] = 2.9451
        elif rh[column][i]*100 == 69:
            frh['f_S'][i] = 2.22
            frh['f_L'][i] = 1.92
            frh['f_SS'][i] = 3.0105
        elif rh[column][i]*100 == 70:
            frh['f_S'][i] = 2.26
            frh['f_L'][i] = 1.95
            frh['f_SS'][i] = 3.0485
        elif rh[column][i]*100 == 71:
            frh['f_S'][i] = 2.31
            frh['f_L'][i] = 1.98
            frh['f_SS'][i] = 3.1269
        elif rh[column][i]*100 == 72:
            frh['f_S'][i] = 2.36
            frh['f_L'][i] = 2.01
            frh['f_SS'][i] = 3.1729
        elif rh[column][i]*100 == 73:
            frh['f_S'][i] = 2.41
            frh['f_L'][i] = 2.05
            frh['f_SS'][i] = 3.2055
        elif rh[column][i]*100 == 74:
            frh['f_S'][i] = 2.47
            frh['f_L'][i] = 2.09
            frh['f_SS'][i] = 3.2459
        elif rh[column][i]*100 == 75:
            frh['f_S'][i] = 2.54
            frh['f_L'][i] = 2.13
            frh['f_SS'][i] = 3.2673
        elif rh[column][i]*100 == 76:
            frh['f_S'][i] = 2.60
            frh['f_L'][i] = 2.18
            frh['f_SS'][i] = 3.3478
        elif rh[column][i]*100 == 77:
            frh['f_S'][i] = 2.67
            frh['f_L'][i] = 2.22
            frh['f_SS'][i] = 3.4174
        elif rh[column][i]*100 == 78:
            frh['f_S'][i] = 2.75
            frh['f_L'][i] = 2.27
            frh['f_SS'][i] = 3.5202
        elif rh[column][i]*100 == 79:
            frh['f_S'][i] = 2.84
            frh['f_L'][i] = 2.33
            frh['f_SS'][i] = 3.5744
        elif rh[column][i]*100 == 80:
            frh['f_S'][i] = 2.93
            frh['f_L'][i] = 2.39
            frh['f_SS'][i] = 3.6329
        elif rh[column][i]*100 == 81:
            frh['f_S'][i] = 3.03
            frh['f_L'][i] = 2.45
            frh['f_SS'][i] = 3.8905
        elif rh[column][i]*100 == 82:
            frh['f_S'][i] = 3.16
            frh['f_L'][i] = 2.52
            frh['f_SS'][i] = 3.8080
        elif rh[column][i]*100 == 83:
            frh['f_S'][i] = 3.27
            frh['f_L'][i] = 2.60
            frh['f_SS'][i] = 3.9505
        elif rh[column][i]*100 == 84:
            frh['f_S'][i] = 3.42
            frh['f_L'][i] = 2.69
            frh['f_SS'][i] = 4.0398
        elif rh[column][i]*100 == 85:
            frh['f_S'][i] = 3.58
            frh['f_L'][i] = 2.79
            frh['f_SS'][i] = 4.1127
        elif rh[column][i]*100 == 86:
            frh['f_S'][i] = 3.76
            frh['f_L'][i] = 2.90
            frh['f_SS'][i] = 4.2824
        elif rh[column][i]*100 == 87:
            frh['f_S'][i] = 3.98
            frh['f_L'][i] = 3.02
            frh['f_SS'][i] = 4.4940
        elif rh[column][i]*100 == 88:
            frh['f_S'][i] = 4.23
            frh['f_L'][i] = 3.16
            frh['f_SS'][i] = 4.6078
        elif rh[column][i]*100 == 89:
            frh['f_S'][i] = 4.53
            frh['f_L'][i] = 3.33
            frh['f_SS'][i] = 4.8573
        elif rh[column][i]*100 == 90:
            frh['f_S'][i] = 4.90
            frh['f_L'][i] = 3.53
            frh['f_SS'][i] = 5.1165
        elif rh[column][i]*100 == 91:
            frh['f_S'][i] = 5.35
            frh['f_L'][i] = 3.77
            frh['f_SS'][i] = 5.3844
        elif rh[column][i]*100 == 92:
            frh['f_S'][i] = 5.93
            frh['f_L'][i] = 4.06
            frh['f_SS'][i] = 5.7457
        elif rh[column][i]*100 == 93:
            frh['f_S'][i] = 6.71
            frh['f_L'][i] = 4.43
            frh['f_SS'][i] = 6.1704
        elif rh[column][i]*100 == 94:
            frh['f_S'][i] = 7.78
            frh['f_L'][i] = 4.92
            frh['f_SS'][i] = 6.7178
        elif 95 <= rh[column][i]*100 <= 100:
            frh['f_S'][i] = 9.34
            frh['f_L'][i] = 5.57
            frh['f_SS'][i] = 7.3492

    return frh
