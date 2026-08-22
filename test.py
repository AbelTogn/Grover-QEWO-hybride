from main import charger_et_predire_image

image_a_tester = "./chien_test.jpg"
classe_devinée = charger_et_predire_image(image_a_tester, "modele_mes_images.pkl", taille_image=(4, 4))

print(f"L'image {image_a_tester} a été identifiée comme : {classe_devinée}")