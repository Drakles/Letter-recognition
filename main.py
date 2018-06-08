from predict import predict
import time
import pickle as pkl


def average_error(predicted, true):
    return sum(predicted != true) / len(predicted)


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step


if __name__ == "__main__":
    start_time = time.time()
    data = pkl.load(open('train.pkl', mode='rb'))
    startIndex = 29000
    predicted = predict(data[0][startIndex:])
    print("average error: ", average_error(predicted, data[1][startIndex:]))
    print("Execution time: ", time.time() - start_time)


def getBestSize():
    best = []
    for i in my_range(6000, 7000, 50):
        actual_exec_time = time.time()
        predicted = predict(data[0][startIndex:], i)
        error = average_error(predicted, data[1][startIndex:])
        if len(best) < 2 or best[1] > error:
            best = (i, error)
            print("actual best= ", best, "execution time:", time.time() - actual_exec_time, " time:",
                  time.time() - start_time)
    print("Execution time: ", time.time() - start_time)
    return best
