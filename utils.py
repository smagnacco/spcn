"""
Utilidades de analisis para el experimento de continual learning del SPCN.

specialization_index / population_specialization implementan una metrica
de selectividad por unidad inspirada en el z-score de selectividad de
features linguisticos usado en Cai et al. (Nature 2026, "Mapping the
neuronal building blocks of human language with language models"):
  Z_i = (mu_i - mu_other) / sigma_other
En ese paper Z_i mide cuanto dispara una neurona para UN feature linguistico
especifico relativo al resto de features. Aqui adaptamos la misma forma
funcional para medir cuanto dispara una unidad del SPCN para UNA task
relativo a otra, en el contexto de continual learning (no hay afirmacion
de que el paper mida esto directamente — es una adaptacion del mismo
principio de selectividad poblacional esparsa a nuestro caso).
"""

import numpy as np


def specialization_index(act_t1, act_t2):
    """Indice de especializacion por unidad entre dos tasks.

    act_t1, act_t2: arrays (n_samples, n_units) de activaciones de la
    misma capa, recogidas sobre el test set de cada task respectivamente.

    Devuelve un array (n_units,) con SI en [-1, 1]:
      SI ~  1  -> la unidad responde fuertemente a t1 y poco a t2
      SI ~ -1  -> la unidad responde fuertemente a t2 y poco a t1
      SI ~  0  -> sin preferencia (responde igual a ambas, o a ninguna)

    Definicion: SI_i = (mu1_i - mu2_i) / (mu1_i + mu2_i + eps)
    Es una version acotada y simetrica del z-score de selectividad
    Z_i = (mu_i - mu_other) / sigma_other usado en el paper, adaptada para
    no depender de sigma_other (que puede ser ~0 bajo activaciones sparse
    top-k, donde muchas unidades estan en 0 para la mayoria de samples).
    """
    act_t1 = np.asarray(act_t1)
    act_t2 = np.asarray(act_t2)
    mu1 = act_t1.mean(axis=0)
    mu2 = act_t2.mean(axis=0)
    eps = 1e-8
    si = (mu1 - mu2) / (mu1 + mu2 + eps)
    return si


def population_specialization(si, threshold=0.5):
    """Fraccion de unidades cuyo |SI| supera threshold, es decir, unidades
    que muestran preferencia marcada por una task sobre la otra (analogo
    al 8.5% de neuronas con z-score >= 2.0 reportado como 'especializacion
    extrema' en Cai et al., aplicado aqui a SI en vez de z-score).

    si: array (n_units,) devuelto por specialization_index.
    threshold: umbral de |SI| para considerar una unidad "especializada".

    Devuelve un float en [0, 1].
    """
    si = np.asarray(si)
    return float(np.mean(np.abs(si) > threshold))
