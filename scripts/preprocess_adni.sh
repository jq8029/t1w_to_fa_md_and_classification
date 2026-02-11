#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="data"

for r_dir in ad nc mci; do
    base="$TARGET_DIR/$r_dir"
    [ -d "$base" ] || continue

    echo "Processing region: $r_dir"

    for subdir in "$base"/*/; do
        [ -d "$subdir" ] || continue
        echo "  Processing subdir: $subdir"

        find "$subdir" -type f -name "*.nii" -print0 |
        while IFS= read -r -d '' f; do
            echo "    Processing file: $f"

            # upper dir（c1）
            cdir=${f%/*}
            cdir=${cdir##*/}
			subdir_name=$(basename "${subdir%/}")
            fname=${f##*/}

            outdir="$TARGET_DIR/$r_dir/${subdir_name}_${cdir}"
			echo "    create dir: $outdir,  file: $f"

            mkdir -p "$outdir"

            outgz="$outdir/$fname"
            mv "$f" "$outgz"

            robustfov -i "$outgz" -r "$outdir/T1w_cropped.nii.gz"
            bet "$outdir/T1w_cropped.nii.gz" \
                "$outdir/T1w_brain.nii.gz" \
                -R -S -m -f 0.25

            echo "      Run: bet $outdir/T1w_cropped.nii.gz $outdir/T1w_brain.nii.gz -R -S -m -f 0.25"

            rm -f "$outgz"
			rm -f "$outdir/T1w_cropped.nii.gz"
        done
		
		rm -rf "$subdir"
    done
done

echo "All done."