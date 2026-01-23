#!/usr/bin/env python

#
# Copyright (c) 2026 Dmitry Arkhipov (grisumbras@yandex.ru)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#

import argparse
import itertools
import io
import jinja2
import os
import os.path
import shutil
import sys


def main(argv):
    args = parse_args(argv)

    source_dir = os.path.abspath(args.source_dir)
    prefix_map = list(itertools.batched(args.map_prefix or [], 2))

    with open(args.target_info, 'r') as f:
        files = parse_tracefile(f, source_dir, prefix_map)

    with open(args.base_info, 'r') as f:
        files = parse_tracefile(f, source_dir, prefix_map, files)

    with open(args.diff, 'r') as f:
        files = parse_diff(f, files, source_dir, prefix_map)

    dirs = collect_directories(files, source_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    stylesheet = os.path.join(os.path.dirname(__file__), 'boostlook.css')
    shutil.copyfile(stylesheet, os.path.join(args.output_dir, 'style.css'))

    template_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(
            os.path.join(os.path.dirname(__file__), 'templates'),
            'utf-8',
        ),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols'],
    )
    file_template = template_env.get_template('file.html')

    for file in files.values():
        path = os.path.join(args.output_dir, file.output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as output:
            with open(file.path, 'r') as text:
                lines = file.calc(text)
                file_template.stream({
                    'file': file,
                    'lines': lines,
                }).dump(output)

    dir_template = template_env.get_template('dir.html')
    for d in sorted(dirs, reverse=True):
        path = os.path.join(args.output_dir, d.output_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as output:
            d.calc()
            dir_template.stream({
                'file': d,
            }).dump(output)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Generate diff coverage report websites.',
    )
    parser.add_argument(
        '-B',
        '--base-info',
        required=True,
        help=f'Tracefile for the base version',
    )
    parser.add_argument(
        '-T',
        '--target-info',
        required=True,
        help=f'Tracefile for the target version',
    )
    parser.add_argument(
        '-D',
        '--diff',
        required=True,
        help=f'Patch file describing changes between versions',
    )
    parser.add_argument(
        '-S', '--source-dir', required=True, help=f'Source directory',
    )
    parser.add_argument(
        '-O', '--output-dir', default=os.getcwd(), help=f'Output directory',
    )
    parser.add_argument(
        '-P', '--map-prefix', nargs='*', help=f'Output directory',
    )

    return parser.parse_args(argv[1:])


def parse_tracefile(lines, source_dir, prefix_map, files=None):
    is_base = files is not None
    if not is_base:
        files = {}

    cur_file = None
    for l in lines:
        if (not l) or l[0] == '#':
            continue

        parts = l.strip().split(':', 1)
        directive = parts[0]

        if directive in ('TN', 'VER'):
            continue

        if directive == 'SF':
            assert cur_file is None
            cur_file = FileData(fix_path(parts[1], prefix_map), source_dir)

        elif directive == 'FNL':
            assert cur_file is not None

            parts = parts[1].split(',', 2)
            func = FunctionData()
            func.index = parts[0]
            func.start = parts[1]
            if len(parts) > 2:
                func.end = parts[2]

            cur_file.functions.add_function(func)

        elif directive == 'FNA':
            assert cur_file is not None
            cur_file.functions.add_alias(*parts[1].split(',', 2))

        elif directive == 'FN':
            assert cur_file is not None

            parts = parts[1].split(',', 2)
            start = parts[0]
            name = parts[-1]
            end = parts[1] if len(parts) > 2 else None

            cur_file.functions.add_alias(start, None, name, end=end)

        elif directive == 'FNDA':
            assert cur_file is not None
            cur_file.functions.update_count(*parts[1].split(',', 1))

        elif directive == 'FNF':
            assert cur_file is not None
            cur_file.functions.total_count = int(parts[1])

        elif directive == 'FNH':
            assert cur_file is not None
            cur_file.functions.total_hits = int(parts[1])

        # elif directive == 'BRDA':
        #     <line_number>,[<exception>]<block>,<branch>,<taken>

        # elif directive == 'BRF':
        #     <number of branches found>

        # elif directive == 'BRH':
        #     <number of branches hit>

        # elif directive == 'MCDC':
        #     <line_number>,<groupSize>,<sense>,<taken>,<index>,<expression>

        # elif directive == 'MRF':
        #     <number of conditions found>

        # elif directive == 'MRH':
        #     <number of condition hit>

        elif directive == 'DA':
            assert cur_file is not None
            parts = parts[1].split(',', 2)
            cur_file.lines.add_line(int(parts[0]), int(parts[1]))

        elif directive == 'LF':
            assert cur_file is not None
            cur_file.lines.total_count = int(parts[1])

        elif directive == 'LH':
            assert cur_file is not None
            cur_file.lines.total_hits = int(parts[1])

        elif directive == 'end_of_record':
            assert cur_file is not None
            if is_base:
                target_file = files.get(cur_file.path)
                if target_file is None:
                    target_file = FileData(cur_file.path, source_dir)
                    files[target_file.path] = target_file
                target_file.base = cur_file
            else:
                assert cur_file.path not in files
                files[cur_file.path] = cur_file

            cur_file = None

        else:
            raise Exception(f'Unknown tracefile directive: {l}')

    return files


def parse_diff(lines, result, source_dir, prefix_map):
    old_file = None
    new_file = None
    file_data = None
    old_line = -1
    new_line = -1
    old_count = 0
    new_count = 0
    changes = []

    def commit():
        if changes:
            assert file_data
            assert old_count == 0
            assert new_count == 0
            file_data.changes = changes

    prior_preprends = 0
    for l in lines:
        if l.startswith('--- '):
            prior_preprends = 0
            commit()
            changes = []
            old_line = -1
            new_line = -1

            old_file = l.split(' ', 1)[1]
            old_file = old_file.split('\t', 1)[0]
            old_file = fix_path(old_file.rstrip(), prefix_map)

        elif l.startswith('+++ '):
            prior_preprends = 0
            new_file = l.split(' ', 1)[1]
            new_file = new_file.split('\t', 1)[0]
            new_file = fix_path(new_file.rstrip(), prefix_map)
            if old_file != new_file:
                if old_file == '/dev/null':
                    old_file = new_file
                elif new_file == '/dev/null':
                    new_file = old_file
            if old_file != new_file:
                print(
                    'error: diffed files have different names:\n'
                    f'\t{old_file}\n'
                    f'\t{new_file}\n'
                    'Did you forget to map a path prefix?',
                    file=sys.stderr
                )
                sys.exit(1)

            file_data = result.get(new_file)
            if file_data is None:
                file_data = FileData(new_file, source_dir)
                # ignore this file, it doesn't have coverage

        elif l.startswith('@@ '):
            prior_preprends = 0
            _, old_range, new_range, _ = l.split(' ', 3)
            assert old_range[0] == '-'
            assert new_range[0] == '+'
            old_range = [int(x) for x in old_range[1:].split(',')]
            if len(old_range) > 1:
                old_line, old_count = old_range
            else:
                old_line = old_range[0]
                old_count = 1
            new_range = [int(x) for x in new_range[1:].split(',')]
            if len(new_range) > 1:
                new_line, new_count = new_range
            else:
                new_line = new_range[0]
                new_count = 1

        elif l.startswith(' '):
            prior_preprends = 0
            old_line += 1
            new_line += 1
            old_count -= 1
            new_count -= 1

        elif l.startswith('-'):
            prior_preprends += 1
            changes.append(('prepend', new_line, old_line, l[1:]))
            old_line += 1
            old_count -= 1

        elif l.startswith('+'):
            if prior_preprends:
                assert prior_preprends > 0
                change = changes[-prior_preprends]
                assert change[0] == 'prepend'
                changes[-prior_preprends] = (
                    'modify', new_line, change[2], change[3],
                )
                prior_preprends -= 1
            else:
                changes.append(('remove', new_line, old_line))
            new_line += 1
            new_count -= 1

        else:
            prior_preprends = 0

    commit()
    return result


def collect_directories(files, source_dir):
    dirs = {}
    more_to_process = {}
    to_process = files.values()
    while to_process:
        for i in to_process:
            if i.path == source_dir:
                continue

            parent = dirs.get(i.parent.path)
            if parent is None:
                dirs[i.parent.path] = i.parent
                more_to_process[i.parent.path] = i.parent
            else:
                i.parent = parent
            i.parent.add_child(i)
        to_process = more_to_process.values()
        more_to_process = {}

    found = True
    while found:
        found = False
        for d in list(dirs.values()):
            if d.path not in dirs:
                continue
            if d.has_parent and len(d.parent.children) == 1:
                if d.parent.has_parent:
                    grandparent = d.parent.parent
                    grandparent.children.remove(d.parent)
                    grandparent.children.append(d)
                else:
                    grandparent = None
                del dirs[d.parent.path]
                d.parent = grandparent
                found = True

    return dirs.values()


def fix_path(path: str, prefix_map) -> str:
    for src, tgt in prefix_map:
        if path.startswith(src):
            path = tgt + path[len(src):]
            break
    return path


def path_suffix(path: str, source_dir: str) -> str:
    if not path.startswith(source_dir):
        print(
            f'error: file path {path} doesn\'t belong to the source directory'
            f' {source_dir}.\nDid you forget to map a path prefix?',
            file=sys.stderr
        )
        sys.exit(1)
    return path[len(source_dir) + 1:]


class FsItemData():
    def __init__(self, path: str, source_dir: str):
        self.path = os.path.join(source_dir, path)
        self.lines = LineDataSummary()
        self._parent: (None | DirData) = None
        self._source_dir = source_dir

    @property
    def source_path(self):
        return path_suffix(self.path, self._source_dir)

    @property
    def has_parent(self):
        return self._parent is not None

    @property
    def basename(self):
        if self.has_parent:
            prefix = self.parent.source_path
            if prefix:
                prefix += '/'
        else:
            prefix = ''
        return self.source_path[len(prefix):]

    @property
    def ancestors(self):
        result = []
        cur = self
        while cur._parent is not None:
            result.append(cur._parent)
            cur = cur._parent
        return list(reversed(result))

    @property
    def stylesheet(self):
        n = len(self.output_path.split(os.sep)) - 1
        return '../' * n + 'style.css'

    @property
    def parent(self):
        if self._parent is None:
            self._parent = DirData(
                os.path.dirname(self.path), self._source_dir
            )
        return self._parent

    @parent.setter
    def parent(self, val: 'DirData'):
        self._parent = val

    @property
    def output_path(self) -> str:
        raise Exception('Not implemented')

    @property
    def output_suffix(self):
        return os.path.basename(self.output_path)

    def relative_path(self, other):
        other_path = other.output_path.split('/')
        path = self.output_path.split('/')
        while path and other_path and path[0] == other_path[0]:
            path = path[1:]
            other_path = other_path[1:]
        other_path = ['..' for _ in range(len(path) - 1)] + other_path
        return '/'.join(other_path)


class FileData(FsItemData):
    def __init__(self, path: str, source_dir: str):
        super().__init__(path, source_dir)
        self.functions = FunctionDataCollection()
        self.branches = BranchDataCollection()
        self.lines = LineDataCollection()
        self.changes = []
        self.base: (None | FileData) = None

    @property
    def output_path(self):
        return self.source_path + '.html'

    def calc(self, text: io.IOBase):
        assert self.base
        result = []
        diff = DiffedLines(text, self)
        it = iter(diff)
        o_l_coverage = sorted(self.base.lines, key=lambda l: l[0])
        n_l_coverage = sorted(self.lines, key=lambda l: l[0])
        try:
            while True:
                line = next(it)
                old_covered = -1
                new_covered = -1
                if o_l_coverage and o_l_coverage[0][0] == line[0]:
                    old_covered = o_l_coverage[0][1]
                    o_l_coverage = o_l_coverage[1:]
                if n_l_coverage and n_l_coverage[0][0] == line[2]:
                    new_covered = n_l_coverage[0][1]
                    n_l_coverage = n_l_coverage[1:]

                kind = ''
                if line[4] in ('k', 'm'):
                    if old_covered < 0:
                        if new_covered == 0:
                            kind = 'UIC'
                        elif new_covered > 0:
                            kind = 'GIC'
                    elif old_covered == 0:
                        if new_covered < 0:
                            kind = 'EUB'
                        elif new_covered == 0:
                            kind = 'UBC'
                        else:
                            kind = 'GBC'
                    else:
                        if new_covered < 0:
                            kind = 'ECB'
                        elif new_covered == 0:
                            kind = 'LBC'
                        else:
                            kind = 'CBC'
                elif line[4] == 'a':
                    if new_covered == 0:
                        kind = 'UNC'
                    elif new_covered > 0:
                        kind = 'GNC'
                elif line[4] == 'r':
                    if old_covered == 0:
                        kind = 'DUB'
                    elif old_covered > 0:
                        kind = 'DCB'

                if kind:
                    counter = getattr(self.lines, kind, 0)
                    setattr(self.lines, kind, counter + 1)

                result.append((
                    line[0],
                    line[1],
                    old_covered,
                    line[2],
                    line[3],
                    new_covered,
                    line[4],
                    kind
                ))
        except StopIteration:
            return result

    def __lt__(self, other: FsItemData):
        if isinstance(other, DirData):
            return False
        return self.path < other.path


class DirData(FsItemData):
    def __init__(self, path: str, source_dir: str):
        super().__init__(path, source_dir)
        self.children = []

    @property
    def output_path(self):
        if self.has_parent:
            return os.path.join(self.source_path, 'index.html')
        else:
            return 'index.html'


    def add_child(self, file: FsItemData):
        assert file.path.startswith(self.path)
        self.children.append(file)

    def calc(self):
        self.children = sorted(self.children)
        for c in self.children:
            self.lines.total_count += c.lines.total_count
            self.lines.total_hits += c.lines.total_hits
            self.lines.UNC += c.lines.UNC
            self.lines.LBC += c.lines.LBC
            self.lines.UIC += c.lines.UIC
            self.lines.UBC += c.lines.UBC
            self.lines.GBC += c.lines.GBC
            self.lines.GIC += c.lines.GIC
            self.lines.GNC += c.lines.GNC
            self.lines.CBC += c.lines.CBC
            self.lines.EUB += c.lines.EUB
            self.lines.ECB += c.lines.ECB
            self.lines.DUB += c.lines.DUB
            self.lines.DCB += c.lines.DCB

    def __lt__(self, other: FsItemData):
        if isinstance(other, FileData):
            return True
        return self.path < other.path


class FunctionDataCollection(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_count = 0
        self.total_hits = 0

    def add_function(self, func):
        assert func.index not in self
        self[func.index] = func

    def add_alias(self, index, count, name, end=None):
        if count is not None:
            assert index in self.keys()
            self[index].aliases[name] = int(count)
        else:
            func = self.get(index)
            if func is None:
                func = FunctionData()
                func.start = index
                func.end = end
                self[index] = func
            assert func.start == index
            assert func.end == end
            assert name not in func.aliases.keys()
            func.aliases[name] = 0

    def update_count(self, count, name):
        for func in self.values():
            if name in func.aliases:
                assert func.aliases[name] == 0
                func.aliases[name] = int(count)
                return
        assert False


class FunctionData():
    def __init__(self):
        self.index = None
        self.start = None
        self.end = None
        self.aliases = {}


class BranchDataCollection():
    pass


class LineDataSummary():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_count: int = 0
        self.total_hits: int = 0
        self.UNC: int = 0
        self.LBC: int = 0
        self.UIC: int = 0
        self.UBC: int = 0
        self.GBC: int = 0
        self.GIC: int = 0
        self.GNC: int = 0
        self.CBC: int = 0
        self.EUB: int = 0
        self.ECB: int = 0
        self.DUB: int = 0
        self.DCB: int = 0


class LineDataCollection(LineDataSummary, list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_line(self, number, count, _=None):
        self.append((number, int(count)))


class DiffedLines():
    def __init__(self, lines: io.IOBase, data: FileData):
        self.lines = lines
        self.data = data

    def __iter__(self):
        changes = sorted(self.data.changes, key=lambda l: l[1])

        it = iter(self.lines)
        n_i = 1
        o_i = 1
        try:
            line = next(it)
            while True:
                if changes and changes[0][1] == n_i:
                    c = changes[0]
                    changes = changes[1:]
                    if c[0] == 'modify':
                        yield o_i, c[3], n_i, line, 'm'
                        n_i += 1
                        o_i += 1
                    elif c[0] == 'remove':
                        yield None, '', n_i, line, 'a'
                        n_i += 1
                    elif c[0] == 'prepend':
                        yield o_i, c[3], None, '', 'r'
                        o_i += 1
                        continue
                    else:
                        assert False
                elif changes and (changes[0][1] < n_i or changes[0][2] < n_i):
                    raise Exception(str(changes))
                else:
                    yield o_i, line, n_i, line, 'k'
                    n_i += 1
                    o_i += 1
                line = next(it)
        except StopIteration:
            pass


if __name__ == '__main__':
    main(sys.argv)
