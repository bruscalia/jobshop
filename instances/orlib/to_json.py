import json


def elements_from_line(line: str):
    machines = []
    p = []
    line.replace("  ", " ")
    elem = line.split()
    N = len(elem) // 2
    for i in range(N):
        machines.append(int(elem[2 * i]))
        p.append(float(elem[2 * i + 1]))
    return machines, p


def txt_to_json(filename: str):
    seq = []
    times = []
    name = filename.replace(".txt", ".json")
    with open(filename, mode="r") as file:
        for j, line in enumerate(file):
            machines, p = elements_from_line(line)
            seq.append(machines)
            times.append(p)
    data = {"seq": seq, "p_times": times}
    with open(name, mode="w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    
    for problem in [
        "abz5", "abz6", "abz7", "abz8", "abz9", "mt06", "mt10", "la01", "la02", "la03",
        "la04", "la05", "la06", "la07", "la08", "la09", "la10", "la11", "la12", "la13", "la14", "la15"
    ]:
        filename = "./../instances/orlib/" + problem + ".txt"
        txt_to_json(filename)