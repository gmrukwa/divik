Simple Windows Instruction
==========================

This is the simplest instruction to run DiviK on Windows.

#. Install Docker (see `Install <./running_in_docker.html#prerequisites>`_)
#. Create ``run_divik.bat`` with following content:

.. literalinclude:: run_divik.bat
    :linenos:
    :language: bat

#. Put your data into ``data.csv``
#. Create ``divik.json`` starting from such template:

.. literalinclude:: ../../divik/_cli/divik.json
    :linenos:
    :language: json

#. Adjust the configuration to your needs

.. note:: Configuration follows the JSON format with fields defined as
``here <https://github.com/gmrukwa/divik/blob/master/divik/_cli/divik.md>`_.

#. Double click the ``run_divik.bat``
