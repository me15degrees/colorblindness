import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab, deltaE_ciede2000
import os

def calculate_iou(mask1, mask2):
    """Calcula a métrica Intersection over Union (IoU) para duas máscaras."""
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 1.0  # Se não houver união, significa que ambas as máscaras estão vazias e são idênticas.
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# --- CONFIGURAÇÃO ---
base_path = 'content'
# Filtros de daltonismo que esperamos encontrar
filter_suffixes = {
    "deuteranopia": "_deuteranopia.png",
    "protanopia": "_protanopia.png",
    "tritanopia": "_tritanopia.png"
}

# Definição do intervalo de cor vermelha para o PTV
lower_red_rgb = np.array([150, 0, 0])
upper_red_rgb = np.array([255, 100, 100])

print("Iniciando análise das imagens...\n")

# --- LOOP PRINCIPAL DE ANÁLISE ---
# Lista todos os arquivos na pasta 'content'
all_files = os.listdir(base_path)

# Encontra as imagens originais (aquelas que não contêm sufixos de filtro)
original_images_names = []
for filename in all_files:
    if filename.endswith(".png") and not any(s in filename for s in filter_suffixes.values()):
        original_images_names.append(filename)

# Garante uma ordem consistente
original_images_names.sort()

if not original_images_names:
    print(f"Nenhuma imagem original (.png) encontrada na pasta '{base_path}'.")
else:
    for original_name in original_images_names:
        original_path = os.path.join(base_path, original_name)
        
        print(f"{'='*50}")
        print(f"=== ANALISANDO IMAGEM ORIGINAL: {original_name} ===")
        print(f"{'='*50}")

        original_image = cv2.imread(original_path)
        if original_image is None:
            print(f"ERRO: Não foi possível carregar a imagem original: {original_path}")
            continue

        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Cria a máscara PTV para a imagem original
        original_ptv_mask = cv2.inRange(original_image_rgb, lower_red_rgb, upper_red_rgb)
        
        # Calcula a cor média do PTV na imagem original (necessário para Delta E)
        # Verifica se a máscara não está vazia para evitar erro de divisão por zero
        if np.sum(original_ptv_mask) > 0:
            mean_color_original_bgr = cv2.mean(original_image, mask=original_ptv_mask)[:3]
            mean_color_original_rgb = np.array([mean_color_original_bgr[2], mean_color_original_bgr[1], mean_color_original_bgr[0]])
            lab_original = rgb2lab(np.uint8([[mean_color_original_rgb]]))
        else:
            print(f"AVISO: Máscara PTV vazia para {original_name}. Delta E não será calculado para esta imagem.")
            mean_color_original_bgr = None # Indica que não há cor para calcular

        # Remove a extensão .png para formar o nome base dos arquivos filtrados
        base_name_without_ext = os.path.splitext(original_name)[0]

        found_filtered_for_original = False
        for filter_type, suffix in filter_suffixes.items():
            filtered_filename = f"{base_name_without_ext}{suffix}"
            filtered_path = os.path.join(base_path, filtered_filename)
            
            print(f"\n--- Comparando com filtro: {filter_type.upper()} ---")

            filtered_image = cv2.imread(filtered_path)

            if filtered_image is None:
                print(f"  -> AVISO: Imagem filtrada não encontrada: {filtered_path}. Pulando cálculos para este filtro.")
                continue
            
            found_filtered_for_original = True

            filtered_image_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
            filtered_gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
            
            # --- 1. Cálculo de IoU ---
            filtered_ptv_mask = cv2.inRange(filtered_image_rgb, lower_red_rgb, upper_red_rgb)
            iou_value = calculate_iou(original_ptv_mask, filtered_ptv_mask)
            print(f"  -> Pontuação de Sobreposição (IoU): {iou_value:.4f}")

            # --- 2. Cálculo de SSIM ---
            ssim_value, _ = ssim(original_gray, filtered_gray, full=True)
            print(f"  -> Índice de Similaridade Estrutural (SSIM): {ssim_value:.4f}")

            # --- 3. Cálculo do Delta E ---
            if mean_color_original_bgr is not None and np.sum(original_ptv_mask) > 0:
                mean_color_filtered_bgr = cv2.mean(filtered_image, mask=original_ptv_mask)[:3]
                mean_color_filtered_rgb = np.array([mean_color_filtered_bgr[2], mean_color_filtered_bgr[1], mean_color_filtered_bgr[0]])
                lab_filtered = rgb2lab(np.uint8([[mean_color_filtered_rgb]]))
                delta_e_value = deltaE_ciede2000(lab_original, lab_filtered)[0][0]
                print(f"  -> Diferença de Cor (Delta E 2000) no PTV: {delta_e_value:.2f}")
            else:
                print(f"  -> Delta E 2000: Não calculado devido à máscara PTV vazia ou cor original não definida.")

        if not found_filtered_for_original:
            print(f"Nenhuma imagem filtrada correspondente encontrada para {original_name}.")
        
print("\nAnálise concluída!")