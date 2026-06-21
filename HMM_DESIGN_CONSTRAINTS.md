# HMM Regime Model — Design Constraints (Anti-Overfitting)

## Objetivo
Crear un modelo de régimen que detecte dinámicamente cuándo:
- **Régimen Calmado**: Vol realizada < implícita (prima rica) → shorting gamma rentable
- **Régimen Turbulento**: Vol realizada > implícita (prima agotada) → long gamma rentable

## 4 GUARDAS CONTRA OVERFITTING

### 1. NO Look-Ahead: Walk-Forward / Causal

**Regla:**
- El régimen en el día `t` se infiere **solo con datos hasta `t`** (inclusive).
- Prohibido: Viterbi suavizado sobre todo el histórico (mirar al futuro).
- Prohibido: Ajustar HMM sobre toda la historia, luego "decidir" regímenes in-sample.

**Implementación:**
- Usar **Forward Algorithm** (causal, probabilidad filtrada), NO Backward/Viterbi.
- O: **Walk-forward**: Entrenar HMM en ventana rolling (ej. 2 años), predecir 1 mes ahead, rotar.
- Mínimo: Reajustar HMM periódicamente (cada trimestre) con datos pasados, aplicar a futuro.

**Test de Sanidad:**
```python
# Pseudo-código
for date_t in dates:
    regime_t = hmm.predict_regime(data_until_t)  # Solo datos hasta t, nunca después
    exposure_t = get_exposure(regime_t)
    pnl_t = execute(exposure_t, data_at_t)
```

---

### 2. Pocas Features, Sentido Económico

**Regla:**
- Máximo 3 features. Punto. Nada de 20.
- Cada feature debe responder una pregunta económica clara.

**Features Propuestas:**
1. **Nivel de VIX** (absoluto): ¿Está la prima implícita cara o barata?
   - Rango histórico: ~10-80. Normalizar a [0, 1].

2. **Vol Realizada Reciente** (ej. 5-20 días): ¿Hay turbulencia ahora?
   - Close-to-close o Parkinson. Comparar contra VIX.

3. _(Opcional)_ **Pendiente Término VIX** (VIX5d - VIX30d): ¿El mercado se calma o se agita?
   - Señal de transición de régimen.

**Prohibido:**
- 20 variables predictoras.
- Variables no económicas (ej. indicadores técnicos puros sin fundamento).

---

### 3. Mapeo Régimen → Exposición por Lógica, NO Optimización

**Regla:**
- El mapeo (régimen) → (exposición) se decide **por lógica económica**, no por maximizar Sharpe del backtest.
- Prohibido: "Deja que el optimizador elija qué régimen compra/vende para maximizar retorno".

**Mapeo Propuesto (Economía Primero):**
```
Régimen CALMADO (Low Realized / High Implied):
  → Exposición CORTA gamma (vender volatilidad)
  → Razón: Prima rica, realizada baja → se espera reversión

Régimen TURBULENTO (High Realized / Low Implied):
  → Exposición LARGA gamma (comprar volatilidad)
  → Razón: Prima agotada, realizada alta → se espera ganancia

Régimen TRANSICIÓN:
  → Exposición PLANA (o reducida)
  → Razón: Incertidumbre, mínimo trades, máximo costes
```

**No hacer:**
```python
# ✗ PROHIBIDO
optimal_mapping = optimize_sharpe(regime_to_exposure, backtest_data)
```

---

### 4. Riesgo Real: Lag en Transiciones (Feb-2018, Mar-2020 Test)

**Regla:**
- El HMM reacciona **tarde** a cambios de régimen.
- Short gamma en régimen "calmado" explotará justo cuando la calma se rompa.
- **Métrica de éxito:** Evaluar out-of-sample, neto de costes. Preguntar crítica:

**Preguntas Clave:**
1. ¿Estaba el modelo **short** (exposición negativa) justo **antes** de Feb-5-2018?
   - Si SÍ → Modelo falló (se la pegó en la transición).
   - Si NO → Modelo fue cauteloso (o suerte).

2. ¿Estaba el modelo **short** justo **antes** de Mar-9-2020?
   - Idem.

3. ¿El modelo cambió a **long** durante el crash (Mar-9 a Mar-20) o después?
   - Si durante → Capturó la ganancia.
   - Si después → Perdió oportunidad y se la pegó.

**Implementación:**
```python
# Post-analysis
regime_long = hmm.regime_at(date='2018-02-03')  # 2 días antes del spike
if regime_long == 'CALM':
    print("⚠️ WARNING: Short exposure before Feb-2018 spike")
    
# Idem para Mar-2020, Mar-2022, etc.
```

---

## Métricas de Evaluación

1. **Out-of-Sample Sharpe** (walk-forward, nunca look-ahead)
2. **Max Drawdown** en regime de transición
3. **Ganancia / Pérdida** específicamente en Feb-2018 y Mar-2020
4. **% de días** que el modelo estuvo short justo antes de crashes
5. **Ratio Costes / P&L Bruto** (para validar si los costes de rolling matan la estrategia)

---

## Resumen: Antes de Escribir Código

✓ Test de Sanidad: Régimen en `t` usa datos solo hasta `t`.
✓ Features: 3 máximo, con sentido económico.
✓ Mapeo: Lógica económica primero, optimization segunda (si acaso).
✓ Riesgo: ¿Short antes de crashes? ¿Lag en transiciones?

**Si el modelo pasa estos filtros, then lo implementamos. Si no, es overfitting bonito.**
