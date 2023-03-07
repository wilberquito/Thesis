import uvicorn
import logging
import vision

def main():
    logging.basicConfig(filename='myapp.log', level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
    vision.run()
    uvicorn.run("api:app", host="0.0.0.0", port=8081, reload=True)


if __name__ == '__main__':
    main()
