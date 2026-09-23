// Choix du nombre de threads OpenMP.
//
// Mesure sur i7-11800H (8 coeurs, 16 threads logiques), 1 h simulee a dt = 10 s,
// configurations alternees pour annuler la derive thermique :
//
//   threads    propagation   grille   paires   total
//         1        0.587 s   0.470 s  2.609 s  3.67 s
//         8        0.156 s   0.509 s  0.437 s  1.10 s
//        16        0.120 s   0.511 s  0.301 s  0.93 s
//
// L'hyper-threading aide (4.9x sur la propagation, 8.7x sur les paires) : le
// defaut suit donc le nombre de threads logiques, celui d'OpenMP. La grille,
// elle, est sequentielle et ne gagne rien — c'est devenu le facteur limitant.

#pragma once

namespace kessler {

// Fixe le nombre de threads OpenMP et renvoie la valeur retenue.
//   requested > 0  : cette valeur
//   requested <= 0 : le defaut d'OpenMP, qu'OMP_NUM_THREADS peut fixer
int configure_threads(int requested);

}  // namespace kessler
