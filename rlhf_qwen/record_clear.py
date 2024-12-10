import os


def get_history_commands():
    history_output = os.popen('bash -c "history"').read()
    # history_output = os.system('history').
    print(history_output)
    lines = history_output.split('\n')
    for line in lines:
        parts = line.split(' ', 1)
        if len(parts) == 2:
            id, command = parts
            print(f"ID: {id}, Command: {command}")


if __name__ == "__main__":

    get_history_commands()