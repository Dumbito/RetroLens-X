import time
import cv2

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print("Backend:", cap.get(cv2.CAP_PROP_BACKEND))
print("FOURCC:", int(cap.get(cv2.CAP_PROP_FOURCC)))
print("Resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Buffer:", cap.get(cv2.CAP_PROP_BUFFERSIZE))
times=[]
start=time.perf_counter()
for _ in range(150):
    t=time.perf_counter()
    ok, frame=cap.read()
    dt=time.perf_counter()-t
    if not ok: break
    times.append(dt)
elapsed=time.perf_counter()-start
cap.release()
print("Frames:", len(times))
print("Elapsed:", f"{elapsed:.2f}s")
print("Effective FPS:", f"{len(times)/elapsed:.2f}")
print("Average read:", f"{sum(times)/len(times)*1000:.2f} ms")
print("Reads <40ms:", sum(t < 0.040 for t in times), "/", len(times))
print("Reads >=80ms:", sum(t >= 0.080 for t in times), "/", len(times))
