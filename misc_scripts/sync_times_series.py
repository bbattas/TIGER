import argparse
from decimal import Decimal, getcontext

getcontext().prec = 50

def format_value(val):
    d = Decimal(val).normalize()
    formatted = format(d, 'f')
    return formatted

def generate_time_series(t_final, n_steps, per_row=16):
    dt = Decimal(str(t_final)) / Decimal(str(n_steps))
    values = []
    for i in range(n_steps + 1):
        val = (dt * i).normalize()
        values.append(format(val, 'f'))

    rows = []
    for i in range(0, len(values), per_row):
        rows.append(' '.join(values[i:i + per_row]))
    print('\n'.join(rows))

def main():
    parser = argparse.ArgumentParser(description='Generate evenly spaced time values for sync_times input.')
    parser.add_argument('-t', '--time', type=float, required=True, help='Final time value')
    parser.add_argument('-n', '--steps', type=int, required=True, help='Number of time steps')
    parser.add_argument('-r', '--per-row', type=int, default=16, help='Entries per row (default: 16)')
    args = parser.parse_args()

    generate_time_series(args.time, args.steps, args.per_row)

if __name__ == '__main__':
    main()
