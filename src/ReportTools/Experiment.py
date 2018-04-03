import functools
from datetime import datetime as datetime
import os

class Experiment(object):
    """
        Базовый класс для всех вычислительных экспериментов.
        Предполагается, что в наследниках должны переопределяться
        методы save и execute.
    """

    # Директория в которую сохраняются результаты экспериментов
    path = os.path.abspath(os.getcwd() + "../../../ExperimentResults/Experiment/")

    def __init__(self, name, reported=True):
        """
            Конструктор.
            name - название эксперимента.
            reported определяет обзателен ли отчет.
        """
        self.name = name
        self.description = input("Write description to experiment: ")
        self.datetime = datetime.today()
        self.report = None
        self.result = None
        if (reported):
            self.save = self._report_decorator(self.save)

    @property
    def Description(self):
        """Описание эксперимента"""
        return self.description

    @property
    def Name(self):
        """Название эксперимента"""
        return self.name

    @property
    def Datetime(self):
        """
        Дата и время проведения эсперимента.
        Объект класса datetime.datetime
        """
        return self.datetime

    Report = property()

    @Report.getter
    def Report(self):
        """Отчет об эксперименте. Текст."""
        return self.report

    @Report.setter
    def Report(self):
        self.report = input("Write report to experiment: ")

    @property
    def Result(self):
        """Результат эксперимента"""
        return self.result

    def _report_decorator(self, func):
        """
        Декоратор, который может быть применен к save.
        В зависимости от того, требуется ли обязательный отчет или нет.
        #todo в инит это задавать и делать явно. Пусть пользователь
        решает хочет отчет или нет.
        """
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if (self.report is None):
                self.report = input("Write report to experiment: ")
            return func(*args, **kwargs)
        return wrapped

    def save(self):
        """
        Метод сохраняет полученный результат.
        В базовом классе это Описание и Отчет.
        #todo работу с файлами посмотри
        """
        path_to_dir = self.__class__.path
        date = str(self.Datetime.timestamp())
        with open(path_to_dir + self.name + ":" + date + ".txt", 'w') as f:
            f.write("Name:" + self.Name + "\n")
            f.write("Description:" + self.Description + "\n")
            f.write("Report:" + self.Report + "\n")

    def _execute_decorator(func):
        """
        Декоратор, который должен быть применен к execute.
        Отвечает за автоматическое присвоение свойству self.result результата
        execute.
        """
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            self.result = func(self, *args, **kwargs)
            return self.result
        return wrapped

    @_execute_decorator
    def execute(self, *args, **kwargs):
        """
        Выполняет эксперимент
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def __str__(self):
        """
        Строковое представление
        :return: string
        """
        return f'Description:\n{self.description}\nDateTime: {self.datetime}'
