---
name: tfce-average-plot-overlay
description: "For multi-event TFCE analyses, the \"average across events\" plot must overlay per-event curves with significant TFCE periods shaded, not a pooled-driver matrix"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 941f2e6e-4278-4a2f-937d-b717cec0928a
---

## Rule

When the user asks for TFCE on multi-event analyses, the "average across events" plot should show the per-event accuracy/statistic curves overlaid, with significant TFCE periods shaded as overlay bands. Do NOT default to producing a pooled-driver matrix (averaging features per uid across events) as the "averaged" view.

## Why

User said: "i dont like it. instead plot the average differences over time. Remember for future, when I ask for the TFCE, i want the average plot to be an overlay of the significant periods". They prefer averaged curves with overlaid significance bands over pooled-driver event-collapsed matrices.

## How to apply

For multi-event TFCE outputs in this FPCA project (and similar), the per-event TFCE results stand on their own; the "averaged" figure should overlay the per-event curves and shade the TFCE-significant time periods rather than recomputing TFCE on a pooled-driver matrix. If the user wants pooled-driver, they will ask explicitly.
