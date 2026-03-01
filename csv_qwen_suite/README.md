# CSV Qwen Suite (ComfyUI Custom Nodes)

Node: **CSV Qwen Prompt + Seed Iterator**
- Loads `positive` / `negative` prompts from a CSV
- Repeats each row `batch_per_row` times
- Advances row/take on every queued execution
- Outputs:
  - `positive`, `negative`
  - `seed` (changes per run)
  - `row_index`, `take_index`, `row_count`
  - `row_id` / `filename_prefix` helper for saving filenames
  - `debug_text` (optional)

Seed formula:
`seed = base_seed + row_index*row_stride + take_index*take_stride`

## Install
Unzip into: `ComfyUI/custom_nodes/`

You should have:
`ComfyUI/custom_nodes/csv_qwen_suite/`

Restart ComfyUI.

## CSV format
Must include a header row. Typical:

```csv
positive,negative
"some positive","some negative"
```

If your headers are different, set:
- `positive_column = <your positive header>`
- `negative_column = <your negative header>`

## Saving with row_id / filename_prefix
This node outputs `row_id` (string), designed to be used as a filename prefix.

Example:
`Face_LoRA_ONLY_r0002_t0_s3985`

### KJNodes (Save Image KJ)
If your save node has an input like `filename_prefix`, wire:
- `row_id` → `Save Image KJ.filename_prefix`

### ComfyUI default Save Image
If your save node has a `filename_prefix` widget or input, use the same.

## Resetting between queues
If your ComfyUI build doesn’t expose a run index in hidden context, this node uses a small file-backed counter.
To restart from row 0 for a new queue:
- set `reset_counter=true` for one run, then turn it back off.
