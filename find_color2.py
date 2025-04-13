import cv2
import numpy as np


# Configuração das cores (intervalo) e gravidades
cores = {
    "azul": ([100, 0, 0], [255, 80, 80]),
    "verde": ([0, 100, 0], [130, 255, 50]),
    "marrom": ([0, 50, 50], [50, 100, 100]),
    "roxo": ([100, 0, 100], [255, 50, 255]),
    "amarelo": ([0, 200, 200], [50, 255, 255]),
    "laranja": ([0, 100, 200], [50, 150, 255]),
    "vermelho": ([0, 0, 100], [95, 85, 255])
}

gravidade = {
    "azul": 0.0,
    "verde": 0.3,
    "marrom": 0.4,
    "roxo": 0.5,
    "amarelo": 0.6,
    "laranja": 0.8,
    "vermelho": 1.0
}

def calcular_gravidade(cor):
    return gravidade.get(cor, 0.0)

def delta_e_cie2000_numpy(lab1, lab2):
    # Convert to float
    L1, a1, b1 = lab1.astype(np.float32)
    L2, a2, b2 = lab2.astype(np.float32)

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    avg_Cp = (C1p + C2p) / 2.0

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    delta_hp = h2p - h1p
    if delta_hp > 180:
        delta_hp -= 360
    elif delta_hp < -180:
        delta_hp += 360

    delta_Hp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(delta_hp / 2.0))
    delta_Lp = L2 - L1
    delta_Cp = C2p - C1p

    avg_Hp = (h1p + h2p) / 2.0
    if abs(h1p - h2p) > 180:
        avg_Hp += 180
    avg_Hp %= 360

    T = 1 - 0.17 * np.cos(np.radians(avg_Hp - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_Hp)) + \
        0.32 * np.cos(np.radians(3 * avg_Hp + 6)) - \
        0.20 * np.cos(np.radians(4 * avg_Hp - 63))

    delta_ro = 30 * np.exp(-((avg_Hp - 275) / 25)**2)
    Rc = 2 * np.sqrt((avg_Cp**7) / (avg_Cp**7 + 25**7))
    Sl = 1 + ((0.015 * (avg_L - 50)**2) / np.sqrt(20 + (avg_L - 50)**2))
    Sc = 1 + 0.045 * avg_Cp
    Sh = 1 + 0.015 * avg_Cp * T
    Rt = -np.sin(np.radians(2 * delta_ro)) * Rc

    delta_E = np.sqrt(
        (delta_Lp / Sl)**2 +
        (delta_Cp / Sc)**2 +
        (delta_Hp / Sh)**2 +
        Rt * (delta_Cp / Sc) * (delta_Hp / Sh)
    )

    return delta_E

# Lista de imagens e filtros
imagens = ("content/0.png", "content/1.png", "content/2.png")
filtros = ("deuteranopia", "protanopia", "tritanopia")

for imagem_original_path in imagens:
    for filtro in filtros:
        nome_base = imagem_original_path.split("/")[-1].split(".")[0]
        imagem_filtrada_path = f"content/{nome_base}_{filtro}.png"

        original_bgr = cv2.imread(imagem_original_path)
        filtrada_bgr = cv2.imread(imagem_filtrada_path)

        if original_bgr is None or filtrada_bgr is None:
            print(f"Erro ao carregar imagens: {imagem_original_path} ou {imagem_filtrada_path}")
            continue

        original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        filtrada = cv2.cvtColor(filtrada_bgr, cv2.COLOR_BGR2RGB)

        if original.shape != filtrada.shape:
            print(f"Tamanhos diferentes: {imagem_original_path} vs {imagem_filtrada_path}")
            continue

        delta_e_total = 0
        erro_gravidade_total = 0
        diferenca_cinza_total = 0
        pixels_vermelhos = 0

        original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        filtrada_lab = cv2.cvtColor(filtrada, cv2.COLOR_RGB2LAB)
        original_cinza = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        filtrada_cinza = cv2.cvtColor(filtrada, cv2.COLOR_RGB2GRAY)

        for y in range(original.shape[0]):
            for x in range(original.shape[1]):
                cor_orig = cor_filt = None
                for cor, (min_val, max_val) in cores.items():
                    if np.all(original[y, x] >= min_val) and np.all(original[y, x] <= max_val):
                        cor_orig = cor
                    if np.all(filtrada[y, x] >= min_val) and np.all(filtrada[y, x] <= max_val):
                        cor_filt = cor
                
                if cor_orig and cor_filt:
                    delta_e = delta_e_cie2000_numpy(original_lab[y, x], filtrada_lab[y, x])
                    delta_e_total += delta_e

                    g_orig = calcular_gravidade(cor_orig)
                    g_filt = calcular_gravidade(cor_filt)
                    erro_gravidade_total += abs(g_orig - g_filt)

                    if cor_orig == "vermelho":
                        diferenca_cinza_total += abs(int(original_cinza[y, x]) - int(filtrada_cinza[y, x]))
                        pixels_vermelhos += 1

        total_pixels = original.shape[0] * original.shape[1]
        print(f"\nResultados para {nome_base}.png com filtro {filtro}:")
        print(f"Métrica de Distorção de Cor (ΔE2000): {delta_e_total / total_pixels:.2f}")
        print(f"Erro Médio de Gravidade: {erro_gravidade_total / total_pixels:.4f}")
        if pixels_vermelhos > 0:
            print(f"MAE em Áreas Críticas (Vermelho): {diferenca_cinza_total / pixels_vermelhos:.2f}")
        else:
            print("Nenhum pixel vermelho detectado na imagem original.")
