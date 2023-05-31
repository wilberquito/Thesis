import logging

import uvicorn


def main():

    logging.basicConfig(filename='myapp.log',
                        level=logging.INFO,
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    uvicorn.run("api.api:app",
                host="0.0.0.0",
                port=8081,
                reload=True)


if __name__ == '__main__':
    main()
