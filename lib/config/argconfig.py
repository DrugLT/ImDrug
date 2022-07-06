import argparse
import inspect
import types
from abc import ABCMeta, abstractmethod
from argparse import Namespace, ArgumentParser, _ArgumentGroup
from json import load
from typing import Any, Dict, Tuple, List, Optional, Union, Sequence, Callable, Type


class Config(metaclass=ABCMeta):
    """
    The basic config class for models, modules, and dadaloaders
    The designer: The one who design the Config for a corresponding class
    The user: The one who call the Config and corresponding class.
    parser name: The name of argument in parser that the config defined as default. The designer should write.
    custom parser name: the user could change the default parser name to their own name when initialize the Config.
    custom default name: the user could change the default "default value" to their own name when initialize the Config.
    kwarg names: The kwarg names of corresponding class. It could be different from parser name.
                 The designer should write.
    """

    def __init__(self, mapping: Optional[Dict[str, Optional[str]]] = None,
                 default: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize config
        Should be written by designer and used by user
        :param mapping: Dict. The mapping for parser (param name --> custom parser name).
                        If parser name --> None, the parser will not add to global parser.
        :param default: Dict. The mapping for the default values of parsers (parser name --> custom default value)
        """
        if mapping is None:
            self.dict = {}
        else:
            self.dict = mapping

        if default is None:
            self.default = {}
        else:
            self.default = default

    def __repr__(self) -> str:
        res = "The parser mapping:\n" + \
              str(self.dict) + "\n" + \
              "The default value mapping:\n" + \
              str(self.default)
        return res

    @abstractmethod
    def get_parser(self, parser: Union[ArgumentParser, _ArgumentGroup, None] = None):
        """
        The function to generate parser. Need to be override by child class.
        Should be written by designer and used by user
        :param parser: The parser passed in. Create new empty parser if None.
        :return: The parser with new arguments.
        """
        if parser is None:
            parser = argparse.ArgumentParser()
        if "config_json" not in [_.dest for _ in parser._actions]:
            parser.add_argument("--config_json",
                                type=str, default=None,
                                help="The config file as json, overwrite the stdin arguments")
        return parser

    @abstractmethod
    def get_config(self, config: Namespace, params: Optional[Dict[str, Any]] = None) -> Tuple[
        List[Any], Dict[str, Any]]:
        """
        Get the args and kwargs for corresponding class. Need to be override by child class
        Should be written by designer and used by user
        :param config: the parse_args() results from argparse
        :param params: Dict. Some values need to be reset instead of the values in config. (parser name --> new value)
                       Note that it will overwrite the value in the config.
        :return args: The args for corresponding class.
        :return kwargs: The kwargs in Dict for corresponding class.
        """

        raise NotImplementedError

    def _parse_config(self,
                      config: Namespace,
                      args: Optional[Sequence[str]],
                      kwargs: Dict[str, str],
                      params: Union[Dict[str, Any]]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        It is a config helper to parser for get_config. Gather values form config and params.
        Should be written by designer and used by user
        :param config: the parse_args() results from argparse
        :param args: The List of parser names for corresponding class. Should be in the order of corresponding class.
        :param kwargs: The Dict of parser names for corresponding kwargs. (kwarg names --> parser names)
        :param params: Dict. Some values need to be reset instead of the values in config. (parser name --> new value)
                       Note that it will overwrite the value in the config.
        :return args: The args for corresponding class.
        :return kwargs: The kwargs in Dict for corresponding class.
        """

        # Parse arguments from json
        if 'config_json' in config and config.config_json is not None:
            config_json = load(open(config.config_json, 'r'))
        else:
            config_json = dict()

        config = vars(config)
        config.update(config_json)
        print(config)

        if args is not None:
            args = self._add_args(config, args, params)
        else:
            args = list()
        kwargs = self._add_kwargs(config, kwargs, params)
        return args, kwargs

    def add_argument(self, parser: Union[ArgumentParser, _ArgumentGroup],
                     arg_name: str, **kwargs) -> ArgumentParser:
        """
        It is a helper for get_parser. It calls argparse.ArgumentParser().add_argument.
        If the argument (custom parser name) already in parser, it will share the same argument.
        If the argument (parser name) in default mapping, it will change the default value.
        Use for designer but is not available for user.
        :param parser: The parser to add the arguments
        :param arg_name: the name of argument. No '--' in the front.
        :param kwargs: The other kwargs for argparse.ArgumentParser().add_argument.
        :return parser: The parser with new argument.
        """
        arg_name = arg_name.lstrip("-")
        # I think we can not add the arg_name which not in self.dict to self.dict here.
        if arg_name not in self.dict:
            self.dict[arg_name] = arg_name
        in_parser = False
        # We can remove this part, since add_argument its self can add the default vale.
        if arg_name in self.default:
            kwargs['default'] = self.default[arg_name]
            # If `default` is set. `required` should be False.
            if 'required' in kwargs:
                kwargs['required'] = False

        for act in parser._actions:
            if act.dest == self.dict[arg_name] or \
                    self.dict[arg_name] in map(lambda s: s.lstrip('-'), act.option_strings):
                in_parser = True
                if "help" in kwargs:
                    # For the duplicate arguments and the previous one do not have help.
                    if act.help is None:
                        act.help = kwargs["help"]
                    else:
                        act.help += ";\n" + kwargs["help"]
                break

        if not in_parser and self.dict[arg_name] is not None:
            parser.add_argument("--" + self.dict[arg_name], **kwargs)

        return parser

    def _add_args(self,
                  config: Union[Namespace, Dict],
                  arg_names: Sequence[str],
                  params: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Get args from the config.
        Use for designer but is not available for user.
        :param config: the parse_args() results
        :param arg_names: the List of parser names for args. Should be in the order of the args of corresponding class
        :param params: Dict. Some values need to be reset instead of the values in config. (parser name --> new value)
                       Usually passed from _parse_config.
        :return: List of arg values
        """
        if isinstance(config, Namespace):
            config = vars(config)
        if params is None:
            params = dict()
        for arg_name in arg_names:
            if arg_name not in self.dict:
                self.dict[arg_name] = arg_name
        # return params if arg_name in params, else return mapping name of self.dict[_] in config, else return None
        return [params.get(_, config.get(self.dict[_], None)) if self.dict[_] is not None
                else params.get(_, None) for _ in arg_names]

    def _add_kwargs(self, config: Union[Namespace, Dict],
                    kwarg_names: Dict[str, str],
                    params: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, Any]:
        """
        Get kwargs from the config.
        Use for designer but is not available for user.
        :param config: the parse_args() results
        :param kwarg_names: The Dict of parser names for corresponding kwargs. (kwarg names --> parser names)
        :param params: Dict. Some values need to be reset instead of the values in config. (parser name --> new value)
                       Usually passed from _parse_config.
        :return: kwargs values in Dict (kwarg name --> value)
        """

        if isinstance(config, Namespace):
            config = vars(config)
        if params is None:
            params = dict()
        for _, config_name in kwarg_names.items():
            if config_name not in self.dict:
                self.dict[config_name] = config_name
            # Should we really need this if statement?
            if self.dict[config_name] is not None and self.dict[config_name] not in config:
                print(f'warn: miss {self.dict[config_name]} in config.')

        return {kwarg_name: params.get(kwarg_name,
                                       config.get(self.dict[config_name], None)
                                       if self.dict[config_name] is not None else None)
                for kwarg_name, config_name in kwarg_names.items()
                if kwarg_name in params or self.dict[config_name] in config}


class SimpleConfig(Config):
    """
    The simple version of Config.
    """

    def __init__(self, mapping: Dict[str, Optional[str]], instance: Union[Callable, Type] = object,
                 ignore_necessary=False, **kwargs):
        """
        Init function
        :param mapping: the mapping dict for (param_name->argument_name).
        :param instance: the target instance to initialize.
        :param **kwargs: the other necessary parameters.
        """

        func = instance if isinstance(instance, types.FunctionType) \
                           or isinstance(instance, types.MethodType) \
            else instance.__init__
        no_default_params, defaults_params = SimpleConfig.get_parameter_name_set(func)
        self.legal_params = no_default_params | defaults_params
        necessary_params = set()
        for p in no_default_params:
            if p not in mapping.keys():
                necessary_params.add(p)
        self.instance = func
        self.ignored_params = defaults_params - mapping.keys()
        if not ignore_necessary:
            self.necessary_params = necessary_params
        else:
            self.necessary_params = set()
            # print("Warning: The necessary param is ignored! You need handle them by yourself.")
        # Use for the case that some necessary_params is not initialized for this config but some child config.
        # For example, the `encoder_out_dim` in `MolPredictor`
        for param_name in self.legal_params:
            if param_name not in mapping:
                mapping[param_name] = param_name

        super(SimpleConfig, self).__init__(mapping)

    def __repr__(self) -> str:
        """
        __repr__function
        :return:
        """
        res = "Target Class: %s\n" % self.instance.__name__ \
              + "The mapped parameters:\n" + \
              str(self.dict.keys()) + "\n" + \
              "The necessary parameters:\n" + \
              str(self.necessary_params) + "\n" + \
              "The ignored_params parameters:\n" + \
              str(self.ignored_params)
        return res

    def _parse_config(self,
                      config: Namespace,
                      args: Optional[Sequence[str]] = None,
                      kwargs: Optional[Dict[str, str]] = None,
                      params: Optional[Dict[str, Any]] = None,
                      ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        The simple version of _parse_config. For internal use only.
        :param config: the target args generated by parse_args.
        :param args:  Deprecated. Do not use.
        :param kwargs: Deprecated. Do not use.
        :param params: Deprecated. Do not use.
        :return:
        """
        assert args is None
        assert kwargs is None
        assert params is None
        return list(), self._add_kwargs(config=config,
                                        kwarg_names=dict(zip(self.dict.keys(), self.dict.keys())),
                                        params=params)

    def get_config(self,
                   args: Namespace,
                   params: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
        """
        Get the config from args.
        :param args: the input arguments.
        :param params: the necessary parameters to initialize target class.
        :return: 
        """
        if params is None:
            params = dict()
        _, kwargs = self._parse_config(args)
        kwargs.update(params)
        for k in self.necessary_params:
            if k not in kwargs:
                raise KeyError("%s is not in necessary_params." % k)
        return kwargs

    def add_argument(self, parser: Union[ArgumentParser, _ArgumentGroup], arg_name: str, **kwargs) -> ArgumentParser:
        """
        The same as Config.add_argument but add argument check whether the argument name is in the list of class
        :param parser: The parser to add the arguments
        :param arg_name: the name of argument. No '--' in the front.
        :param kwargs: The other kwargs for argparse.ArgumentParser().add_argument.
        :return parser: The parser with new argument.
        """
        if arg_name.lstrip("-") not in self.legal_params:
            raise ValueError(f'The argument {arg_name.lstrip("-")} should be in the argument list {self.legal_params}.')

        return super(SimpleConfig, self).add_argument(parser, arg_name, **kwargs)

    @classmethod
    def get_parameter_name_set(cls, func, except_list=("self", "cls")):
        """
        The function to get the parameter set
        :param func: The target function.
        :param except_list: the excepted parameter name.
        :return:
        """
        args = inspect.getfullargspec(func).args
        defaults = inspect.getfullargspec(func).defaults
        # remove none case.
        defaults = set() if defaults is None else defaults
        no_default_params = set(args[:(len(args) - len(defaults))]) - set(except_list)
        defaults_params = set(args[(len(args) - len(defaults)):]) - set(except_list)

        return no_default_params, defaults_params


class Configurable(metaclass=ABCMeta):
    """The interface for the configurable object"""

    @classmethod
    @abstractmethod
    def get_config_class(cls):
        pass

    @classmethod
    @abstractmethod
    def get_instance(cls, config: Config, args, params=None):
        pass
 