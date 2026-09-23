# Índices espectrais × CRA

Unidade experimental: média das oito leituras espectrais matinais por bloco × genótipo × condição, pareada ao CRA da mesma parcela.
Datas pareadas: D03M–24/02, D09M–02/03 e D10M–03/03. O conjunto possui 71 parcelas válidas.

Os resultados por data são a evidência principal. A linha com todas as datas é apenas descritiva, pois pode incorporar o efeito simultâneo do tempo em CRA e índice.

| Data CRA | Índice | n | ρ Spearman | p Spearman | r Pearson | p Pearson | R² linear |
|---|---|---:|---:|---:|---:|---:|---:|
| 24/02 | WBI | 24 | 0.460 | 0.0237 | 0.460 | 0.0236 | 0.212 |
| 24/02 | SRWI | 24 | 0.212 | 0.3196 | 0.275 | 0.1936 | 0.076 |
| 24/02 | NDWI | 24 | 0.212 | 0.3196 | 0.276 | 0.1916 | 0.076 |
| 24/02 | Clorofila a (razão) | 24 | 0.656 | 0.0005 | 0.600 | 0.0019 | 0.360 |
| 24/02 | Clorofila b (razão) | 24 | 0.677 | 0.0003 | 0.619 | 0.0013 | 0.383 |
| 24/02 | MCARI (clorofila total) | 24 | -0.679 | 0.0003 | -0.561 | 0.0043 | 0.315 |
| 02/03 | WBI | 23 | 0.325 | 0.1301 | 0.417 | 0.0478 | 0.174 |
| 02/03 | SRWI | 23 | 0.117 | 0.5962 | 0.186 | 0.3947 | 0.035 |
| 02/03 | NDWI | 23 | 0.117 | 0.5962 | 0.187 | 0.3923 | 0.035 |
| 02/03 | Clorofila a (razão) | 23 | 0.817 | 0.0000 | 0.916 | 0.0000 | 0.838 |
| 02/03 | Clorofila b (razão) | 23 | 0.851 | 0.0000 | 0.921 | 0.0000 | 0.848 |
| 02/03 | MCARI (clorofila total) | 23 | -0.667 | 0.0005 | -0.720 | 0.0001 | 0.519 |
| 03/03 | WBI | 24 | 0.138 | 0.5194 | 0.134 | 0.5318 | 0.018 |
| 03/03 | SRWI | 24 | 0.035 | 0.8718 | -0.081 | 0.7060 | 0.007 |
| 03/03 | NDWI | 24 | 0.035 | 0.8718 | -0.080 | 0.7106 | 0.006 |
| 03/03 | Clorofila a (razão) | 24 | 0.277 | 0.1909 | 0.246 | 0.2474 | 0.060 |
| 03/03 | Clorofila b (razão) | 24 | 0.183 | 0.3931 | 0.160 | 0.4550 | 0.026 |
| 03/03 | MCARI (clorofila total) | 24 | -0.335 | 0.1098 | -0.237 | 0.2656 | 0.056 |
| Todas as datas (descritivo) | WBI | 71 | 0.234 | 0.0493 | 0.236 | 0.0475 | 0.056 |
| Todas as datas (descritivo) | SRWI | 71 | -0.377 | 0.0012 | -0.366 | 0.0017 | 0.134 |
| Todas as datas (descritivo) | NDWI | 71 | -0.377 | 0.0012 | -0.365 | 0.0018 | 0.133 |
| Todas as datas (descritivo) | Clorofila a (razão) | 71 | 0.449 | 0.0001 | 0.489 | 0.0000 | 0.239 |
| Todas as datas (descritivo) | Clorofila b (razão) | 71 | 0.463 | 0.0000 | 0.511 | 0.0000 | 0.261 |
| Todas as datas (descritivo) | MCARI (clorofila total) | 71 | -0.459 | 0.0001 | -0.510 | 0.0000 | 0.260 |

## Fórmulas

- **WBI:** `R900 / R970` — Índice de água foliar.
- **SRWI:** `R860 / R1240` — Índice de água de razão simples.
- **NDWI:** `(R860 − R1240) / (R860 + R1240)` — Índice normalizado de água.
- **Clorofila a (razão):** `R430 / R622` — Proxy espectral de clorofila a.
- **Clorofila b (razão):** `R453 / R642` — Proxy espectral de clorofila b.
- **MCARI (clorofila total):** `[(R700 − R670) − 0,2 × (R700 − R550)] × (R700 / R670)` — Índice de absorção de clorofila total.
