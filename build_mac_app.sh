#!/bin/bash
# Build a thin .app wrapper for StreamDiffusion GUI entry points.
# The resulting .app lives in the project root and launches the existing
# .venv + Python scripts; it does NOT bundle the whole environment.
#
# Usage:
#   ./build_mac_app.sh --ndi        # default; builds StreamDiffusion-NDI-GUI.app
#   ./build_mac_app.sh --rgbd       # builds StreamDiffusion-RGBD-GUI.app
#   ./build_mac_app.sh --rgbd-db    # builds StreamDiffusion-RGBD-GUI-DB.app

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE="ndi"
for arg in "$@"; do
    case "$arg" in
        --ndi)       MODE="ndi" ;;
        --rgbd)      MODE="rgbd" ;;
        --rgbd-db)   MODE="rgbd-db" ;;
        --help|-h)
            echo "Usage: $0 [--ndi | --rgbd | --rgbd-db]"
            exit 0
            ;;
    esac
done

case "$MODE" in
    ndi)
        APP_NAME="StreamDiffusion-NDI-GUI.app"
        SCRIPT="python/stream_ndi_gui.py"
        BUNDLE_ID="com.streamdiffusion.ndi-gui"
        DISPLAY_NAME="StreamDiffusion-NDI-GUI"
        ;;
    rgbd)
        APP_NAME="StreamDiffusion-RGBD-GUI.app"
        SCRIPT="python/stream_rgbd_gui.py"
        BUNDLE_ID="com.streamdiffusion.rgbd-gui"
        DISPLAY_NAME="StreamDiffusion-RGBD-GUI"
        ;;
    rgbd-db)
        APP_NAME="StreamDiffusion-RGBD-GUI-DB.app"
        SCRIPT="python/stream_rgbd_gui_db.py"
        BUNDLE_ID="com.streamdiffusion.rgbd-gui-db"
        DISPLAY_NAME="StreamDiffusion-RGBD-GUI-DB"
        ;;
esac

TEMPLATE="$PROJECT_DIR/build/mac_app/AppTemplate.app"
DST_APP="$PROJECT_DIR/$APP_NAME"

if [ ! -d "$TEMPLATE" ]; then
    echo "ERROR: App template not found at $TEMPLATE"
    exit 1
fi

echo "Building $APP_NAME ..."

# Remove old copy and place the app in the project root.
rm -rf "$DST_APP"
cp -R "$TEMPLATE" "$DST_APP"

# Update the plist for the selected entry point.
PLIST="$DST_APP/Contents/Info.plist"
sed -i '' "s|com.streamdiffusion.ndi-gui|$BUNDLE_ID|g" "$PLIST"
sed -i '' "s|StreamDiffusion-NDI-GUI|$DISPLAY_NAME|g" "$PLIST"

# Write the launcher with the correct script path.
LAUNCHER="$DST_APP/Contents/MacOS/launcher"
cat > "$LAUNCHER" <<EOF
#!/bin/bash
# StreamDiffusion GUI launcher for macOS .app bundle.
# This is a thin wrapper: the .app must live in the project root,
# and the .venv virtual environment must already exist.
set -e

APP_DIR="\$(cd "\$(dirname "\$0")/.." && pwd)"
PROJECT_DIR="\$(cd "\$APP_DIR/../../.." && pwd)"
VENV="\$PROJECT_DIR/.venv/bin/activate"
SCRIPT="\$PROJECT_DIR/$SCRIPT"

if [ ! -f "\$VENV" ]; then
    osascript -e "display alert \\"Environment not found\\" message \\"Could not find the virtual environment at \$VENV. Please run python/setup.sh first.\\"" >&2
    exit 1
fi

if [ ! -f "\$SCRIPT" ]; then
    osascript -e "display alert \\"Script not found\\" message \\"Could not find \$SCRIPT. Is this .app placed in the project root?\\"" >&2
    exit 1
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "\$PROJECT_DIR"

source "\$VENV"
exec python "\$SCRIPT" "\$@"
EOF

chmod +x "$LAUNCHER"

echo "Created: $DST_APP"
echo ""
echo "Usage:"
echo "  1. Make sure .venv exists (run python/setup.sh if needed)."
echo "  2. Double-click $APP_NAME in Finder, or run:"
echo "     open '$DST_APP'"
echo ""
echo "To add a custom icon, replace:"
echo "  $DST_APP/Contents/Resources/app_icon.icns"
echo "and add CFBundleIconFile to:"
echo "  $DST_APP/Contents/Info.plist"
