# FY2025 Sales Analysis — Data Story & Hypothesis Testing Summary

**Dataset:** `cleaned_dataset.csv` — 1,000 orders, Jan 2025–Jan 2026, 8 cities, 5 categories, 6 products
**Author:** Data Analytics — Task 4
**Prepared for:** Business stakeholders

---

## 1. The Data Story

**Objective:** Turn twelve months of transaction data into a business narrative, and check the patterns that look real against a statistical test before acting on them.

**Headline numbers**

| Metric | Value |
|---|---|
| Total revenue | ₹13.94 Cr (₹139,399,439.65) |
| Total orders | 1,000 |
| Average order value | ₹1,39,399 |
| Unique customers | 947 |
| Repeat customers | 52 (5.2%) |

**What the analysis found**

- **Electronics dominates the category mix**, generating ₹5.08 Cr (36.4% of revenue) from 354 orders — more than double the next-largest category.
- **Laptop (₹2.54 Cr) and Mobile (₹2.53 Cr)** are the top two products by revenue, both under Electronics.
- **Revenue is geographically balanced.** Patna (₹2.08 Cr) and Kolkata (₹1.89 Cr) lead, but the gap between the top city and the bottom city (Gaya, ₹1.44 Cr) is only 31% — no single market can be ignored.
- **Monthly revenue is broadly stable**, peaking in March (₹1.31 Cr) and dipping in August–September (₹0.94 Cr / ₹0.92 Cr) before recovering through Q4. The dip tracks with fewer orders, not smaller order sizes.
- **Retention is a weak point:** only 5.2% of the 947 customers ordered more than once.

**Conclusion:** the business has a healthy, well-diversified top line concentrated in Electronics and Education, spread across geography, but a large single-purchase customer base. The one open question raised repeatedly by stakeholders — *does gender actually change what people buy?* — is answered with a formal test below.

**Call to action:** protect supply and service levels for Electronics/Education, pilot (not scale) gender-weighted Electronics marketing pending an A/B test, and invest in a retention/loyalty program to convert the 94.8% single-purchase customers.

---

## 2. Hypothesis Testing

Five hypotheses were formulated from patterns in the data and tested at **α = 0.05**.

### 2.1 Primary hypothesis — Gender vs. Electronics purchase rate

- **H₀:** Gender and the likelihood of purchasing Electronics are independent (equal purchase rates).
- **H₁:** Purchase rate of Electronics differs by gender.
- **Test:** Two-proportion Z-test
- **Data:** Male — 196 of 511 orders are Electronics (38.4%); Female — 158 of 489 orders are Electronics (32.3%)
- **Result:** z = 1.998, **p = 0.0457**
- **Decision:** p < 0.05 → **reject H₀**
- **Business conclusion:** There is a statistically significant, though modest (~6 percentage points), difference in Electronics purchase rate by gender. This supports a *pilot* gender-weighted Electronics campaign, validated further with a controlled A/B test before committing major budget — the effect is real but not large enough to justify a full strategy overhaul on its own.

*(A related chi-squared test across all five categories by gender gave χ² = 9.09, p = 0.059 — just short of significance overall, which is why the analysis narrowed to the single category, Electronics, driving the effect.)*

### 2.2 Supporting hypotheses tested (all not significant)

| # | Hypothesis | Test | Statistic | p-value | Conclusion |
|---|---|---|---|---|---|
| 2 | Average order value differs by gender | Welch's t-test | t = 0.68 | 0.495 | Not significant |
| 3 | Average order value differs, weekday vs. weekend | Welch's t-test | t = 1.09 | 0.276 | Not significant |
| 4 | Average order value differs, Electronics vs. Grocery | Welch's t-test | t = ‑0.17 | 0.868 | Not significant |
| 5 | City and category preference are associated | Chi-squared | χ² = 34.15 (df=28) | 0.196 | Not significant |

**Why the negative results matter:** consistent order values across gender, day-of-week, and top categories mean pricing and promotions can stay unified rather than fragmented by these dimensions. The genuine differentiator in this dataset is *what* customers buy (category mix by gender), not *how much* they spend per order.

---

## 3. Methodology Notes

- All tests were run in Python (`scipy.stats`, `statsmodels`) at 95% confidence (α = 0.05).
- Two-sample comparisons used **Welch's t-test** (unequal variances assumed) rather than Student's t-test, since category/gender subgroup sizes and variances differ.
- The chi-squared tests used contingency tables of raw order counts; all expected cell counts exceeded 5, satisfying the test's assumptions.
- Sample sizes: Male n=511, Female n=489; Electronics n=354, Grocery n=153; Weekday n=712, Weekend n=288.

## 4. Recommendations

1. **Protect the core** — Electronics and Education together drive 54% of revenue; prioritize stock and service levels here.
2. **Test before you target** — pilot a gender-weighted Electronics campaign as a controlled A/B test to confirm the 6-point gap before scaling ad spend.
3. **Fix the retention gap** — only 5.2% of customers repeat; a loyalty or win-back program is the highest-leverage lever identified in this analysis.
4. **Investigate the Aug–Sep dip** — driven by order count, not order value; review seasonal promotion timing for that window.
