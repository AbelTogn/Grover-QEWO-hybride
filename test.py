from main import charger_et_predire_image, TAILLE_IMAGE

# il faut que votre image soit dans le même répertoire que ce fichier
image_a_tester = "./image.jpg"
classe_devinée = charger_et_predire_image(image_a_tester, "modele_images.pkl", taille_image=TAILLE_IMAGE)

print(f"L'image {image_a_tester} a été identifiée comme : {classe_devinée}")