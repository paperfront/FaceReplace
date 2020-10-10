from face_write import write_all_faces
import sys

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) != 3:
        print("Proper usage: driver.py [background_path] [source path]")
        return
    background_path, source_path = argv[1:]
    write_all_faces(background_path, source_path)
