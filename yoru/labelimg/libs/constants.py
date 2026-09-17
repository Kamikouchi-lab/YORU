SETTING_FILENAME = 'filename'
SETTING_RECENT_FILES = 'recentFiles'
SETTING_WIN_SIZE = 'window/size'
SETTING_WIN_POSE = 'window/position'
SETTING_WIN_GEOMETRY = 'window/geometry'
SETTING_LINE_COLOR = 'line/color'
SETTING_FILL_COLOR = 'fill/color'
SETTING_ADVANCE_MODE = 'advanced'
SETTING_WIN_STATE = 'window/state'
SETTING_SAVE_DIR = 'savedir'
SETTING_PAINT_LABEL = 'paintlabel'
SETTING_LAST_OPEN_DIR = 'lastOpenDir'
SETTING_AUTO_SAVE = 'autosave'
SETTING_SINGLE_CLASS = 'singleclass'
FORMAT_PASCALVOC='PascalVOC'
FORMAT_YOLO='YOLO'
FORMAT_CREATEML='CreateML'
# YOLO-OBB shares the .txt extension with plain YOLO; the two are told
# apart by field count when loading (see libs/yolo_obb_io.is_obb_file).
FORMAT_YOLO_OBB='YOLO-OBB'
SETTING_DRAW_SQUARE = 'draw/square'
SETTING_LABEL_FILE_FORMAT= 'labelFileFormat'
DEFAULT_ENCODING = 'utf-8'

# Remembered across sessions so that reopening labelImg on an OBB project
# does not silently start writing axis-aligned boxes over it.
SETTING_OBB_MODE = 'obbMode'
