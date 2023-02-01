
root="/work/oxr"

if [[ $* == *-tr* ]]; then
    path="$root/cache/training"
    echo "Removing $path"
    rm -r $path
fi

if [[ $* == *-te* ]]; then
    path="$root/cache/testing"
    echo "Removing $path"
    rm -r $path
fi

echo "Done"
