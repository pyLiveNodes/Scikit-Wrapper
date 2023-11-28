from pathlib import Path
import random
import numpy as np

from livenodes.producer import Producer

from livenodes_core_nodes.ports import Ports_empty, Port_Bool, Port_TS_Number, Port_List_Number, Port_TS_Number, Port_Str, Port_List_Str, Ports_empty
from typing import NamedTuple

import glob

class Ports_out(NamedTuple):
    data: Port_TS_Number = Port_TS_Number("Data")
    channels: Port_List_Str = Port_List_Str("Channel Names")
    time: Port_List_Number = Port_List_Number("Time")
    session: Port_Str = Port_Str("Session")
    trial: Port_Str = Port_Str("Trial")
    retro: Port_Bool = Port_Bool("Retro")

class EASE_read_plux(Producer):
    """
    Reads plux data from the ease tsd dataset.
    Every trial is saved in a batch of the (batch, time, channel) data format.
    I.e. one trial makes up a full emit/ret call 
    """

    ports_in = Ports_empty()
    ports_out = Ports_out()

    category = "Data Source"
    description = ""

    example_init = {
        "sessions": "folder/**",
        "hub": "plux-98",
        "shuffle": False,
        "name": "EASE Plux",
    }

    def __init__(self,
                 sessions,
                 hub="plux-98",
                 shuffle=False,
                 name="EASE Plux",
                 **kwargs):
        super().__init__(name, **kwargs)

        self.sessions = sessions
        self.shuffle = shuffle
        self.hub = hub

        # TODO: change these depending on the selected hub
        if hub == 'plux-98':
            self.channels = ["0", "1", "2", "3", "4", "5", "6", "7"]
        elif hub == 'plux-99':
            self.channels = ["8", "9", "10", "11", "12", "13", "14", "15"]
        else:
            raise ValueError('TSD1 only recorded hubs "plux-98" and "plux-99"')

    def _settings(self):
        return {\
            "sessions": self.sessions,
            "shuffle": self.shuffle,
            "hub": self.hub
        }

    def _name_trial(self, path):
        folder = path.name # only keep last folder
        if 'retro' in str(folder):
            # matches 'trial.' and selects '01' from 'trial.01[.retro]'
            return 't' + str(folder)[6:8] + 'r' + str(folder)[-1:]
        elif 'eyes-closed' in str(folder):
            return 'ec'
        elif 'eyes-open' in str(folder):
            return 'eo'
        return 't' + str(folder)[6:8]

    def _run(self):
        # TODO: we should re-consider the "send once pattern that is currently established for channel names..."
        self.ret_accu(self.channels, port=self.ports_out.channels)

        sessions_list = glob.glob(self.sessions)

        if self.shuffle:
            random.shuffle(sessions_list)
        else:
            sessions_list = list(sorted(sessions_list))

        for session_path in sessions_list:
            session_name = 's' + session_path[-4:-1]

            trials = [f for f in Path(session_path).glob('**/*') if f.is_dir()]
            
            if self.shuffle:
                random.shuffle(trials)
            else:
                trials = list(sorted(trials))
            
            for trial in trials:
                trial_name = self._name_trial(trial)
                retro = 'retro' in str(trial)

                self.debug(f'Reading {session_name}{trial_name}.')

                self.ret_accu(session_name, port=self.ports_out.session)
                self.ret_accu(trial_name, port=self.ports_out.trial)
                self.ret_accu(retro, port=self.ports_out.retro)

                try:
                    # TODO: make this explicit calls to path, rather than this implicit conversion
                    plux98_csv_path = trial/(f'{session_name}{trial_name}.biosignal.{self.hub}.data.csv')
                    plux98_np = np.genfromtxt(plux98_csv_path, delimiter=',')

                    # channel zero is lsl time
                    self.ret_accu(plux98_np[:,0], port=self.ports_out.time)
                    # channels 1-8 are plux channels, todo: name them in the __init__
                    self.ret_accu(np.array([plux98_np[:,1:]]), port=self.ports_out.data)
                    
                    yield self.ret_accumulated()

                except Exception as err:
                    # scrap this, lets just raise an error in the logger and don't sent any data
                    self.logger.error(err)

if __name__ == "__main__":
    node = EASE_read_plux(sessions="/share/data/ease-tsd/publish/session.05*/")
    for g in node._run():
        printable = {key: np.array(val).shape if key in ['data', 'time'] else val for key, val in g.items()}
        print(printable)
