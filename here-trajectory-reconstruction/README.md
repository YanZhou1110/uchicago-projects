# HERE Trajectory Reconstruction

A graduate capstone exploring trajectory reconstruction from monocular driving imagery and noisy GPS observations.

## Yan Zhou's contribution

- Image ingestion, quality diagnostics and feature-matching analysis.
- An ORB visual odometry baseline.
- GPS coordinate conversion and standardized TUM-format trajectory exports for downstream integration.
- Diagnostic visualizations for low-confidence parts of a sequence.

## Scope of this public overview

This page summarizes the work described in Yan's supplied resume and local handoff materials. Client/team source code, raw driving images, GPS trajectories, private documents and geospatial exports are not distributed. No public code release or client endorsement is implied.

The ORB workflow is a baseline, not a production reconstruction system. Its use of GPS-derived scale means agreement with GPS is not independent ground-truth validation. Final capstone results and newer SLAM work are not asserted here.
