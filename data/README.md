# Data Format

No individual-level pediatric research data are distributed with this public
repository. To run the manuscript evaluation, authorized users with approved
data access must supply the following four comma-separated CSV files locally:

- `ASM_features.csv`: one `sample_id` column containing de-identified sample
  IDs, followed by 523 numerical ASM feature columns.
- `FOSM_features.csv`: one `sample_id` column containing the matching
  de-identified sample IDs, followed by 79 numerical FOSM feature columns.
- `labels.csv`: one `sample_id` column and one binary `label` column
  (`0` = non-pneumonia and `1` = pneumonia).
- `fold_indices.csv`: one `sample_id` column and one integer `fold` column,
  with fold indices ranging from 1 to 5.

The four files must contain exactly the same set of de-identified sample IDs.
Each `sample_id` must be unique within its file, all feature values must be
finite numerical values, and all feature-column names must be unique.

The feature matrices, labels, and fold assignments are not publicly provided
because they remain linked to individual-level pediatric research records.
Real feature values, sample IDs, original filenames, fold assignments, and
other participant-related information must not be committed to this repository.