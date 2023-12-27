import cv2


def get_camera_info(camera_id):
    cap = cv2.VideoCapture(camera_id)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {
        "width": width,
        "height": height,
        "fps": fps,
    }


def find_connected_cameras(start=0, end=10):
    connected_cameras = []
    for i in range(start, end + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera ID {i} is connected.")
            info = get_camera_info(i)
            connected_cameras.append((i, info))
        cap.release()
    return connected_cameras


if __name__ == "__main__":
    cameras = find_connected_cameras()
    for id, info in cameras:
        print(
            f"Camera ID: {id}, Width: {info['width']}, Height: {info['height']}, FPS: {info['fps']}"
        )
