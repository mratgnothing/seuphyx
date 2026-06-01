"""
Oil drop experiment tabs module.
Each tab is implemented as a separate function for better maintainability.
"""

from seuphyx.core.oil.tabs.tab_record import render_tab_record
from seuphyx.core.oil.tabs.tab_train import render_tab_train
from seuphyx.core.oil.tabs.tab_classify import render_tab_classify
from seuphyx.core.oil.tabs.tab_regress import render_tab_regress
from seuphyx.core.oil.tabs.tab_vision import render_tab_vision
from seuphyx.core.oil.tabs.tab_report import render_tab_report


__all__ = [
    'render_tab_record',
    'render_tab_train',
    'render_tab_classify',
    'render_tab_regress',
    'render_tab_vision',
    'render_tab_report',
]
