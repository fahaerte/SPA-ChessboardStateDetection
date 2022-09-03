from fastapi import FastAPI
import uvicorn
import ChessboardStateDetectionWithKMeans as kmeans_detection

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run("ChessboardStateDetectionWithUniqueColors:app")


@app.get("/")
async def hello_world():
    return "Send request to endpoint \"chessboardstate\" with an optinoal url-parameter called \"url\" " \
           "which is used to specify the location of an image. If the parameter is missing, " \
           "the current image of the chessboard is retrieved. Only provide images of size 424 x 240."


@app.get("/chessboardstate")
async def get_chessboard_state(url: str):
    return kmeans_detection.calculate_chessboard_state(url)


@app.get("/chessboardstate/")
async def get_chessboard_state(url: str):
    return kmeans_detection.calculate_chessboard_state(url)


@app.get("/current/chessboardstate")
async def get_chessboard_state():
    return kmeans_detection.calculate_chessboard_state()


@app.get("/current/chessboardstate/")
async def get_chessboard_state():
    return kmeans_detection.calculate_chessboard_state()
