"""Visualisation pygame du pendule contrôlé par le réseau."""
import numpy as np

import config as cfg
from pendule import Pendule


def visualiser(nn):
    import pygame  # import tardif : inutile pour entraîner

    pygame.init()
    largeur, hauteur = 900, 700
    ecran = pygame.display.set_mode((largeur, hauteur))
    pygame.display.set_caption(f"Pendule à {cfg.N_SEGMENTS} segments — contrôle neuronal")

    horloge = pygame.time.Clock()
    police = pygame.font.SysFont(None, 24)
    origine = np.array([largeur // 2, 150], dtype=float)
    echelle = min(100.0, 400.0 / max(1, cfg.N_SEGMENTS))

    pendule = Pendule(cfg.N_SEGMENTS)
    pendule.reinitialiser()
    pause = False
    actif = True

    while actif:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actif = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actif = False
                elif event.key == pygame.K_SPACE:
                    pause = not pause
                elif event.key == pygame.K_r:
                    pendule.reinitialiser()

        if not pause:
            commande = nn.predire(pendule.etat())
            pendule.avancer(cfg.COUPLE_MAX * commande)

        ecran.fill((247, 247, 247))
        points = [origine.copy()]
        position = origine.copy()

        for angle in pendule.theta:
            position = position + np.array(
                [echelle * np.sin(angle), echelle * np.cos(angle)]
            )
            points.append(position.copy())

        for i in range(cfg.N_SEGMENTS):
            a = tuple(points[i].astype(int))
            b = tuple(points[i + 1].astype(int))
            pygame.draw.line(ecran, (50, 85, 145), a, b, 7)
            pygame.draw.circle(ecran, (205, 65, 65), b, 10)

        pygame.draw.circle(ecran, (25, 25, 25), tuple(origine.astype(int)), 12)

        ecran.blit(
            police.render(
                "Espace : pause | R : réinitialiser | Échap : quitter",
                True, (20, 20, 20),
            ),
            (20, 20),
        )
        ecran.blit(
            police.render(
                "Angles : " + ", ".join(f"{x:.2f}" for x in pendule.theta),
                True, (20, 20, 20),
            ),
            (20, 50),
        )
        pygame.display.flip()
        horloge.tick(60)

    pygame.quit()
