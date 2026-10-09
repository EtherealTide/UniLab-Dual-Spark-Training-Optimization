# Keep the full stream in the remote log; limit only coordinator console output.
{
    clean = $0
    gsub(/\033\[[0-9;]*[A-Za-z]/, "", clean)
    gsub(/\r/, "", clean)
    if (mode == "full") {
        print "[" label "] " $0
        fflush()
    } else if (clean ~ /Using device:|Synchronizing parameters|Downloading .* from HF|Downloaded directory|\[launcher\]|^ssh:|Permission denied|Host key verification failed/) {
        print "[" label "] " clean
        fflush()
    } else if (match(clean, /(Learning iteration|Iterations:)[[:space:]]+[0-9]+\/[0-9]+/)) {
        progress = substr(clean, RSTART, RLENGTH)
        zero_based = progress ~ /^Learning iteration/
        sub(/^[^0-9]*/, "", progress)
        split(progress, counts, "/")
        completed = counts[1] + (zero_based ? 1 : 0)
        total = counts[2] + 0
        now = systime()
        if (!last_progress || now - last_progress >= interval || (completed == total && progress != previous_progress)) {
            printf "[%s] Iteration %d/%d\n", label, completed, total
            fflush()
            last_progress = now
            previous_progress = progress
        }
    }
}
