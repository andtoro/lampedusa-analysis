import pandas as pd

def func2(dt, rh, column):
    # Initialize the DataFrame
    frh = pd.DataFrame({'time': dt})
    frh['f_S'] = 0.0
    frh['f_L'] = 0.0
    frh['f_SS'] = 0.0

    # Iterate over the rows
    for i in range(len(rh[column])):

        rh_value = round(rh.loc[i, column]*100)  # Extract the value once to avoid repetition
        
        # Conditions based on the value of rh_value
        if 0 <= rh_value <= 36:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.00, 1.00, 1.00]
        elif rh_value == 37:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.38, 1.31, 1.00]
        elif rh_value == 38:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.40, 1.32, 1.00]
        elif rh_value == 39:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.42, 1.34, 1.00]
        elif rh_value == 40:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.44, 1.35, 1.00]
        elif rh_value == 41:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.46, 1.36, 1.00]
        elif rh_value == 42:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.48, 1.38, 1.00]
        elif rh_value == 43:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.49, 1.39, 1.00]
        elif rh_value == 44:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.51, 1.41, 1.00]
        elif rh_value == 45:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.53, 1.42, 1.00]
        elif rh_value == 46:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.55, 1.44, 1.00]
        elif rh_value == 47:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.57, 1.45, 2.3584]
        elif rh_value == 48:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.59, 1.47, 2.3799]
        elif rh_value == 49:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.62, 1.49, 2.4204]
        elif rh_value == 50:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.64, 1.50, 2.4488]
        elif rh_value == 51:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.66, 1.52, 2.4848]
        elif rh_value == 52:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.68, 1.54, 2.5006]
        elif rh_value == 53:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.71, 1.55, 2.5052]
        elif rh_value == 54:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.73, 1.57, 2.5279]
        elif rh_value == 55:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.76, 1.59, 2.5614]
        elif rh_value == 56:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.78, 1.61, 2.5848]
        elif rh_value == 57:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.81, 1.63, 2.5888]
        elif rh_value == 58:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.83, 1.65, 2.6160]
        elif rh_value == 59:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.86, 1.67, 2.6581]
        elif rh_value == 60:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.89, 1.69, 2.6866]
        elif rh_value == 61:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.92, 1.71, 2.7341]
        elif rh_value == 62:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.95, 1.73, 2.7834]
        elif rh_value == 63:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [1.99, 1.75, 2.8272]
        elif rh_value == 64:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.02, 1.78, 2.8287]
        elif rh_value == 65:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.06, 1.80, 2.8594]
        elif rh_value == 66:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.09, 1.83, 2.8943]
        elif rh_value == 67:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.13, 1.86, 2.9105]
        elif rh_value == 68:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.17, 1.89, 2.9451]
        elif rh_value == 69:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.22, 1.92, 3.0105]
        elif rh_value == 70:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.26, 1.95, 3.0485]
        elif rh_value == 71:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.31, 1.98, 3.1269]
        elif rh_value == 72:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.36, 2.01, 3.1729]
        elif rh_value == 73:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.41, 2.05, 3.2055]
        elif rh_value == 74:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.47, 2.09, 3.2459]
        elif rh_value == 75:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.54, 2.13, 3.2673]
        elif rh_value == 76:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.60, 2.18, 3.3478]
        elif rh_value == 77:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.67, 2.22, 3.4174]
        elif rh_value == 78:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.75, 2.27, 3.5202]
        elif rh_value == 79:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.84, 2.33, 3.5744]
        elif rh_value == 80:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [2.93, 2.39, 3.6329]
        elif rh_value == 81:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.03, 2.45, 3.8905]
        elif rh_value == 82:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.16, 2.52, 3.8080]
        elif rh_value == 83:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.27, 2.60, 3.9505]
        elif rh_value == 84:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.42, 2.69, 4.0398]
        elif rh_value == 85:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.58, 2.79, 4.1127]
        elif rh_value == 86:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.76, 2.90, 4.2824]
        elif rh_value == 87:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [3.98, 3.02, 4.4940]
        elif rh_value == 88:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [4.23, 3.16, 4.6078]
        elif rh_value == 89:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [4.53, 3.33, 4.8573]
        elif rh_value == 90:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [4.90, 3.53, 5.1165]
        elif rh_value == 91:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [5.35, 3.77, 5.3844]
        elif rh_value == 92:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [5.93, 4.06, 5.7457]
        elif rh_value == 93:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [6.71, 4.43, 6.1704]
        elif rh_value == 94:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [7.78, 4.92, 6.7178]
        elif 95 <= rh_value <= 100:
            frh.loc[i, ['f_S', 'f_L', 'f_SS']] = [9.34, 5.57, 7.3492]

    return frh
