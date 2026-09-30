"""격자 결과를 복제하지 않고 표와 CSV에 제공하는 행 뷰입니다."""
from dataclasses import dataclass


@dataclass
class GridRows:
    axes: tuple
    fields: tuple
    coordinate_order: tuple = (0, 1)

    def __post_init__(self):
        if len(self.axes) != 2 or any(array.shape != tuple(map(len, self.axes)) for array in self.fields):
            raise ValueError('격자 축과 결과 배열의 크기가 다릅니다.')
        if sorted(self.coordinate_order) != [0, 1]:
            raise ValueError('좌표 열의 순서가 올바르지 않습니다.')

    def __len__(self):
        return len(self.axes[0]) * len(self.axes[1])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        i, j = divmod(index, len(self.axes[1]))
        coordinates = (self.axes[0][i], self.axes[1][j])
        return tuple(coordinates[k] for k in self.coordinate_order) + tuple(array[i, j] for array in self.fields)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
