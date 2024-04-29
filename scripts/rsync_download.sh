SERVER="$1"
TO_DOWNLOAD="$2"
DEST="$3"

rsync -azh -e 'ssh' --info=progress2 $SERVER:$TO_DOWNLOAD $DEST