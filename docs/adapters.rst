Adapters
--------
Included adapters
=================

OpenSCM currently supports the following simple climate models using
adapters:

+--------------------------------------------------------------------+-----------+----------+---------------------------------------------------------------------------------------------------------+
| Model name                                                         | Stepwise? | Regions? | Comment                                                                                                 |
+====================================================================+===========+==========+=========================================================================================================+
| `DICE <https://sites.google.com/site/williamdnordhaus/dice-rice>`_ | yes       | no       | | Climate component from the Dynamic Integrated                                                         |
|                                                                    |           |          | | Climate-Economy (DICE) model by `William Nordhaus <https://sites.google.com/site/williamdnordhaus/>`_ |
+--------------------------------------------------------------------+-----------+----------+---------------------------------------------------------------------------------------------------------+

See :ref:`writing-adapters` on how to implement an adapter for a
further SCM.

Adapter base class
==================

.. automodule:: openscm.adapter
