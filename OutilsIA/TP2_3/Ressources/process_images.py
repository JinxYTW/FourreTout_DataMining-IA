import os
from PIL import Image

# Définition des répertoires
input_folder = "images"  # Dossier contenant les images d'origine avec sous-répertoires
output_folder = "images_small"  # Dossier pour stocker les images redimensionnées
label_file = "labels.txt"  # Fichier de labels

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_folder, exist_ok=True)

# Obtenir une liste triée des sous-dossiers
categories = sorted([d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))])

# Ouvrir le fichier de labels en écriture
with open(label_file, "w") as f:
    # Parcourir les sous-répertoires triés
    for category in categories:
        category_path = os.path.join(input_folder, category)
        
        # Crée une liste triée des fichiers dans le sous-dossier
        files = sorted([file for file in os.listdir(category_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        for filename in files:
            img_path = os.path.join(category_path, filename)
            img = Image.open(img_path)
            img_resized = img.resize((32, 32))
            
            # Modifier le nom du fichier pour éviter les doublons
            new_filename = f"{category}_{filename}"
            output_image_path = os.path.join(output_folder, new_filename)
            img_resized.save(output_image_path)
            
            # Écriture du nom de la catégorie pour chaque image dans le fichier de labels
            f.write(f"{category}\n")

print("Redimensionnement terminé. Les images sont enregistrées dans", output_folder)
print("Fichier de labels créé avec les catégories pour chaque image:", label_file)
